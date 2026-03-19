"""
cheating_classifier.py  —  Two-stage cheating classification.

Stage 1 — Rule-based (always active):
  Slides a 60-second window of events per student.
  Each event type carries a weight. Window total maps to:
    NORMAL  (<25)  /  SUSPICIOUS  (<60)  /  CHEATING  (≥60)

Stage 2 — ML (optional, activates when ≥50 labelled samples exist):
  A scikit-learn RandomForestClassifier trained on feature vectors
  extracted from the same event windows.  Predicts the same three labels.
  Falls back to Stage 1 when the model is unavailable.

Usage:
    from classification.cheating_classifier import CheatingClassifier

    clf = CheatingClassifier()
    clf.record_event("S018", "CALCULATOR_DETECTED", timestamp=time.time())
    result = clf.classify("S018")   # -> Classification(label="CHEATING", ...)
    report = clf.get_all_classifications()
"""
from __future__ import annotations

import logging
import time
import pickle
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config.config import cfg as _cfg
    _WINDOW_SEC    = float(_cfg.classifier.window_seconds)
    _THRESH_SUSP   = float(_cfg.classifier.thresholds.suspicious)
    _THRESH_CHEAT  = float(_cfg.classifier.thresholds.cheating)
    _WEIGHTS: Dict[str, float] = dict(_cfg.classifier.feature_weights)
    _MODEL_PATH    = Path(_cfg.classifier.ml.model_path)
    _MIN_SAMPLES   = int(_cfg.classifier.ml.retrain_min_samples)
except Exception:
    _WINDOW_SEC, _THRESH_SUSP, _THRESH_CHEAT = 60.0, 25.0, 60.0
    _WEIGHTS = {
        "LOOKING_LEFT": 2, "LOOKING_RIGHT": 2, "LOOKING_DOWN": 3,
        "BODY_LEANING": 4, "HANDS_ON_FACE": 3, "TALKING_DETECTED": 5,
        "WHISPERING_DETECTED": 8, "PHONE_DETECTED": 12,
        "LOOKING_AT_PHONE": 10, "CALCULATOR_DETECTED": 15,
        "PASSING_OBJECT": 20, "WRITING_ON_PALM": 18,
        "SEAT_VACATED": 6, "BEHAVIOR_ANOMALY": 4,
        "FACE_ABSENT": 5, "MULTIPLE_PERSONS": 10, "EYE_MOVEMENT": 3,
    }
    _MODEL_PATH  = Path("models/classifier/cheating_clf.pkl")
    _MIN_SAMPLES = 50

_ALL_EVENT_TYPES = sorted(_WEIGHTS.keys())
_LABELS = ["NORMAL", "SUSPICIOUS", "CHEATING"]


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class _Event:
    name:      str
    timestamp: float
    weight:    float = field(init=False)

    def __post_init__(self) -> None:
        self.weight = _WEIGHTS.get(self.name, 2.0)


@dataclass
class Classification:
    sid:          str
    label:        str          # NORMAL / SUSPICIOUS / CHEATING
    rule_score:   float        # raw rule-based score
    ml_label:     Optional[str] = None   # ML prediction (None if model not ready)
    ml_confidence: float = 0.0
    event_counts: Dict[str, int] = field(default_factory=dict)
    window_seconds: float = _WINDOW_SEC


# ── Classifier ────────────────────────────────────────────────────────────────

class CheatingClassifier:
    """
    Per-student cheating classifier with a sliding event window.

    Thread-safety: NOT thread-safe. Call from a single thread
    (the main AI processing thread) or add external locking.
    """

    def __init__(self) -> None:
        # sid -> deque of _Event (auto-expires old events on access)
        self._windows: Dict[str, Deque[_Event]] = defaultdict(deque)
        # Labelled training samples for ML: (feature_vec, label_str)
        self._training: List[Tuple[np.ndarray, str]] = []
        # Fitted sklearn model (None until enough samples)
        self._model = None
        self._load_model()
        log.info("CheatingClassifier ready  (window=%.0fs  thresholds=%.0f/%.0f)",
                 _WINDOW_SEC, _THRESH_SUSP, _THRESH_CHEAT)

    # ── Public API ────────────────────────────────────────────────────────────

    def record_event(self, sid: str, event_name: str,
                     timestamp: Optional[float] = None) -> None:
        """Record one alert event for a student."""
        ts = timestamp if timestamp is not None else time.time()
        self._windows[sid].append(_Event(name=event_name, timestamp=ts))

    def classify(self, sid: str) -> Classification:
        """Classify one student based on their current event window."""
        self._expire(sid)
        window = list(self._windows[sid])
        counts  = self._event_counts(window)
        score   = sum(e.weight for e in window)
        label   = self._rule_label(score)
        ml_lbl, ml_conf = self._ml_predict(self._feature_vec(counts))
        return Classification(
            sid=sid, label=label, rule_score=round(score, 1),
            ml_label=ml_lbl, ml_confidence=ml_conf,
            event_counts=counts,
        )

    def get_all_classifications(self) -> Dict[str, Classification]:
        """Return classifications for every student seen so far."""
        return {sid: self.classify(sid) for sid in self._windows}

    def add_training_sample(self, sid: str, true_label: str) -> None:
        """
        Label the current window of sid as true_label and add to training set.
        Call this when a human invigilator confirms or overrides a classification.
        """
        self._expire(sid)
        counts = self._event_counts(list(self._windows[sid]))
        vec    = self._feature_vec(counts)
        self._training.append((vec, true_label))
        log.debug("Training sample added for %s: %s  (total=%d)",
                  sid, true_label, len(self._training))
        if len(self._training) >= _MIN_SAMPLES:
            self._train_model()

    def reset_student(self, sid: str) -> None:
        self._windows[sid].clear()

    def reset_all(self) -> None:
        self._windows.clear()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _expire(self, sid: str) -> None:
        """Drop events older than the window from the deque."""
        cutoff = time.time() - _WINDOW_SEC
        d = self._windows[sid]
        while d and d[0].timestamp < cutoff:
            d.popleft()

    @staticmethod
    def _event_counts(window: List[_Event]) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for e in window:
            counts[e.name] += 1
        return dict(counts)

    @staticmethod
    def _rule_label(score: float) -> str:
        if score >= _THRESH_CHEAT:
            return "CHEATING"
        if score >= _THRESH_SUSP:
            return "SUSPICIOUS"
        return "NORMAL"

    def _feature_vec(self, counts: Dict[str, int]) -> np.ndarray:
        """Convert event-count dict to a fixed-length feature vector."""
        return np.array(
            [counts.get(e, 0) for e in _ALL_EVENT_TYPES],
            dtype=np.float32,
        )

    # ── ML layer ──────────────────────────────────────────────────────────────

    def _ml_predict(
        self, vec: np.ndarray
    ) -> Tuple[Optional[str], float]:
        if self._model is None:
            return None, 0.0
        try:
            proba = self._model.predict_proba([vec])[0]
            idx   = int(np.argmax(proba))
            label = self._model.classes_[idx]
            return label, round(float(proba[idx]), 3)
        except Exception as e:
            log.warning("ML prediction failed: %s", e)
            return None, 0.0

    def _train_model(self) -> None:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.preprocessing import LabelEncoder

            X = np.array([s[0] for s in self._training])
            y = [s[1] for s in self._training]

            clf = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X, y)
            self._model = clf
            _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_MODEL_PATH, "wb") as f:
                pickle.dump(clf, f)
            log.info("ML classifier trained on %d samples → %s",
                     len(self._training), _MODEL_PATH)
        except Exception as e:
            log.error("ML training failed: %s", e)

    def _load_model(self) -> None:
        if not _MODEL_PATH.exists():
            log.debug("No pre-trained ML model found at %s — rule-based only", _MODEL_PATH)
            return
        try:
            with open(_MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            log.info("ML classifier loaded from %s", _MODEL_PATH)
        except Exception as e:
            log.warning("Could not load ML model: %s — rule-based only", e)
