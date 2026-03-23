"""
auto_threshold.py
=================
Per-student dynamic threshold calibration for the classroom proctoring system.

How it works
------------
During the first few seconds of the exam (default: 3 seconds at ~30 fps = ~90 frames),
every frame is sampled for each student. After the window closes, the collected
samples are analysed per-signal per-student:

  1.  Outliers are removed (IQR filter)
  2.  Median is used as the personal baseline  (robust against blinks / head turns)
  3.  Std-dev is used to auto-size the deadband / threshold offset
  4.  A stability check ensures the variance was low enough to trust the window.
      If variance is too high, the window extends by EXTENSION_SEC seconds until
      it settles or the hard cap is reached.

Signals calibrated
------------------
  - gaze_ratio        (iris gaze 0-1, ~0.5 = centre)
  - look_down_score   (iris-below-lid score, low = normal)
  - lean              (shoulder spread, abs(left.x - right.x))
  - mar               (mouth aspect ratio)
  - head_ratio        (nose.x relative to face width, ~0.5 = centre)

Thresholds produced  (stored in AutoThreshold.thresholds[sid])
-------------------
  For each signal:
    baseline   — personal neutral value (median of calibration window)
    std        — standard deviation across the window
    thr_high   — baseline + k*std  (triggers "too high" alert)
    thr_low    — baseline - k*std  (triggers "too low"  alert)

Integration
-----------
  1.  Create one AutoThreshold instance in ai_engine.py at module level.
  2.  Call  .feed(sid, signals_dict)  on every frame during calibration.
  3.  Call  .finalise(sid)            when you want to lock thresholds for a student.
  4.  Call  .is_ready(sid)            to check if thresholds are locked.
  5.  Use   .get(sid, signal)         to retrieve the threshold dict.
  6.  Use   .apply_to_learner(sid, learner_baseline)  to push values into
       AdaptiveLearner's StudentBaseline so the rest of the pipeline works
       transparently.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AutoThresholdConfig:
    # ── Window settings ────────────────────────────────────────────────────
    fps: float = 30.0            # expected camera fps (used to convert sec → frames)
    window_sec: float = 3.0      # base calibration window in seconds
    extension_sec: float = 1.5   # extend window by this much if variance is too high
    max_window_sec: float = 8.0  # never extend beyond this

    # ── Minimum samples before we even attempt finalisation ────────────────
    min_samples: int = 20        # need at least this many samples per signal

    # ── Stability threshold ────────────────────────────────────────────────
    # If coefficient-of-variation (std/|mean|) > this, window is "unstable"
    cv_stable_threshold: float = 0.25

    # ── Threshold multiplier k (deadband = k × std) ───────────────────────
    # Higher = less sensitive (fewer false positives)
    k_gaze: float = 2.5
    k_look_down: float = 2.0
    k_lean: float = 2.0
    k_mar: float = 2.5
    k_head: float = 2.5

    # ── Absolute minimum deadband per signal (guards against std ≈ 0) ─────
    min_deadband_gaze: float = 0.04
    min_deadband_look_down: float = 0.008
    min_deadband_lean: float = 0.025
    min_deadband_mar: float = 0.012
    min_deadband_head: float = 0.04

    # ── Fallback values (used when calibration fails / window too short) ──
    fallback_gaze_baseline: float = 0.500
    fallback_look_down: float = 0.042
    fallback_lean: float = 0.12
    fallback_mar: float = 0.018
    fallback_head: float = 0.500

    # ── IQR outlier filter ─────────────────────────────────────────────────
    iqr_factor: float = 2.0      # values outside median ± iqr_factor*IQR are dropped

    # ── Debug ─────────────────────────────────────────────────────────────
    debug: bool = True


# ---------------------------------------------------------------------------
# Per-student calibration buffer
# ---------------------------------------------------------------------------

@dataclass
class _StudentCalibBuffer:
    """Accumulates raw per-frame signal values during the calibration window."""
    # Raw sample lists — one entry per sampled frame
    gaze: List[float]       = field(default_factory=list)
    look_down: List[float]  = field(default_factory=list)
    lean: List[float]       = field(default_factory=list)
    mar: List[float]        = field(default_factory=list)
    head: List[float]       = field(default_factory=list)

    window_start: float     = field(default_factory=time.time)
    extended_until: float   = 0.0   # non-zero = window has been extended
    extension_count: int    = 0     # how many times we extended


# ---------------------------------------------------------------------------
# Per-student threshold result
# ---------------------------------------------------------------------------

@dataclass
class SignalThreshold:
    baseline: float   = 0.0
    std: float        = 0.0
    thr_high: float   = 0.0    # baseline + deadband
    thr_low: float    = 0.0    # baseline - deadband (NaN if signal is one-sided)
    is_fallback: bool = False   # True = could not calibrate, using hardcoded default


@dataclass
class StudentThresholds:
    gaze:      SignalThreshold = field(default_factory=SignalThreshold)
    look_down: SignalThreshold = field(default_factory=SignalThreshold)
    lean:      SignalThreshold = field(default_factory=SignalThreshold)
    mar:       SignalThreshold = field(default_factory=SignalThreshold)
    head:      SignalThreshold = field(default_factory=SignalThreshold)

    finalised_at: float = 0.0
    is_fallback: bool   = True   # True until properly calibrated


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AutoThreshold:
    """
    Manages per-student dynamic threshold calibration.

    Typical lifecycle
    -----------------
    Phase A — calibration window (before exam lock):
        at = AutoThreshold()
        for each frame:
            at.feed(sid, {"gaze": g, "look_down": ld, "lean": l, "mar": m, "head": h})
            if at.should_finalise(sid):
                at.finalise(sid)

    Phase B — detection (after exam lock):
        if at.is_ready(sid):
            thr = at.get(sid, "gaze")
            looking_left  = gaze_ratio < thr.thr_low
            looking_right = gaze_ratio > thr.thr_high
    """

    SIGNALS = ("gaze", "look_down", "lean", "mar", "head")

    def __init__(self, config: AutoThresholdConfig = None):
        self.cfg = config or AutoThresholdConfig()
        self._buffers:    Dict[str, _StudentCalibBuffer]  = {}
        self._thresholds: Dict[str, StudentThresholds]    = {}
        self._finalised:  set                             = set()

        self._window_frames = int(self.cfg.fps * self.cfg.window_sec)
        log.info(
            "[AutoThreshold] Init — window=%.1fs (%d frames), k=gaze:%.1f/ld:%.1f/lean:%.1f/mar:%.1f",
            self.cfg.window_sec, self._window_frames,
            self.cfg.k_gaze, self.cfg.k_look_down, self.cfg.k_lean, self.cfg.k_mar,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, sid: str, signals: Dict[str, Optional[float]]):
        """
        Call once per frame per student during the calibration window.

        signals keys: "gaze", "look_down", "lean", "mar", "head"
        Values may be None if the signal was not detected this frame — they
        are silently skipped so partial frames don't corrupt the buffer.
        """
        if sid in self._finalised:
            return   # already locked — nothing to do

        buf = self._get_buf(sid)

        for signal in self.SIGNALS:
            val = signals.get(signal)
            if val is not None and np.isfinite(val):
                getattr(buf, signal).append(float(val))

    def should_finalise(self, sid: str) -> bool:
        """
        Returns True when the student's calibration window is complete
        and it's time to call finalise().

        Logic:
          - base window must be full (min_samples met)
          - if variance is too high, extend by extension_sec (up to max_window_sec)
          - if max window reached, finalise regardless
        """
        if sid in self._finalised:
            return False

        buf = self._get_buf(sid)
        now = time.time()
        elapsed = now - buf.window_start

        # Not enough time elapsed for base window
        if elapsed < self.cfg.window_sec:
            return False

        # Check we have enough samples
        n = self._min_samples(buf)
        if n < self.cfg.min_samples:
            # Not enough samples yet — give more time unless we've hit the hard cap
            if elapsed < self.cfg.max_window_sec:
                return False
            else:
                # Hard cap — finalise with fallback
                log.warning(
                    "[AutoThreshold] %s: hard cap reached with only %d samples — using fallback",
                    sid, n,
                )
                return True

        # Check variance stability
        if not self._is_stable(buf):
            max_window = self.cfg.max_window_sec
            if elapsed < max_window:
                if buf.extension_count == 0 or now > buf.extended_until:
                    buf.extended_until  = now + self.cfg.extension_sec
                    buf.extension_count += 1
                    log.info(
                        "[AutoThreshold] %s: variance high — extending window by %.1fs "
                        "(extension %d, elapsed=%.1fs)",
                        sid, self.cfg.extension_sec, buf.extension_count, elapsed,
                    )
                return False  # still in extended window
            else:
                log.warning(
                    "[AutoThreshold] %s: max window reached with high variance — finalising anyway",
                    sid,
                )

        return True

    def finalise(self, sid: str) -> StudentThresholds:
        """
        Compute and lock thresholds for a student.
        Safe to call multiple times — subsequent calls are no-ops.
        """
        if sid in self._finalised:
            return self._thresholds[sid]

        buf  = self._get_buf(sid)
        thr  = StudentThresholds()
        thr.finalised_at = time.time()

        # ── per-signal computation ──────────────────────────────────────────
        thr.gaze      = self._compute_signal(buf.gaze,      self.cfg.k_gaze,
                                              self.cfg.min_deadband_gaze,
                                              self.cfg.fallback_gaze_baseline, "gaze")
        thr.look_down = self._compute_signal(buf.look_down, self.cfg.k_look_down,
                                              self.cfg.min_deadband_look_down,
                                              self.cfg.fallback_look_down, "look_down")
        thr.lean      = self._compute_signal(buf.lean,      self.cfg.k_lean,
                                              self.cfg.min_deadband_lean,
                                              self.cfg.fallback_lean, "lean")
        thr.mar       = self._compute_signal(buf.mar,       self.cfg.k_mar,
                                              self.cfg.min_deadband_mar,
                                              self.cfg.fallback_mar, "mar")
        thr.head      = self._compute_signal(buf.head,      self.cfg.k_head,
                                              self.cfg.min_deadband_head,
                                              self.cfg.fallback_head, "head")

        # If most signals are fallback, mark the whole profile as fallback
        n_fallback = sum(
            1 for sig in self.SIGNALS if getattr(thr, sig).is_fallback
        )
        thr.is_fallback = n_fallback >= 3

        self._thresholds[sid] = thr
        self._finalised.add(sid)

        if self.cfg.debug:
            self._log_thresholds(sid, thr, buf)

        # Cleanup buffer — free memory
        if sid in self._buffers:
            del self._buffers[sid]

        return thr

    def is_ready(self, sid: str) -> bool:
        """True if this student's thresholds have been finalised."""
        return sid in self._finalised

    def get(self, sid: str, signal: str) -> Optional[SignalThreshold]:
        """
        Return the SignalThreshold for (sid, signal).
        Returns None if not yet finalised.
        """
        thr = self._thresholds.get(sid)
        if thr is None:
            return None
        return getattr(thr, signal, None)

    def get_all(self, sid: str) -> Optional[StudentThresholds]:
        return self._thresholds.get(sid)

    def apply_to_learner(self, sid: str, baseline) -> bool:
        """
        Push auto-calibrated values into an AdaptiveLearner StudentBaseline.

        Parameters
        ----------
        sid       : student ID string
        baseline  : a StudentBaseline dataclass instance from adaptive_learning.py

        Returns True on success, False if not yet finalised.
        """
        if not self.is_ready(sid):
            return False

        thr = self._thresholds[sid]

        # yaw baseline → gaze neutral point
        baseline.yaw       = thr.gaze.baseline
        # look_down baseline
        baseline.look_down = thr.look_down.baseline
        # lean baseline
        baseline.lean      = thr.lean.baseline
        # mar baseline
        baseline.mar       = thr.mar.baseline

        # Force mark as calibrated by bumping sample counts
        # (the AdaptiveLearner uses sample_count >= CALIBRATION_FRAMES to decide)
        from adaptive_learning import Config as LearnerConfig
        cfg = LearnerConfig()
        baseline.sample_count      = max(baseline.sample_count,      cfg.CALIBRATION_FRAMES)
        baseline.lean_sample_count = max(baseline.lean_sample_count, cfg.CALIBRATION_FRAMES_LEAN)
        baseline.last_updated      = time.time()

        log.info(
            "[AutoThreshold] Applied to AdaptiveLearner baseline for %s: "
            "yaw=%.4f ld=%.4f lean=%.4f mar=%.4f",
            sid, baseline.yaw, baseline.look_down, baseline.lean, baseline.mar,
        )
        return True

    def get_thresholds_for_learner_config(self, sid: str) -> Optional[Dict[str, float]]:
        """
        Returns a dict of REL_* override values tuned for this student,
        suitable for replacing the global Config values in AdaptiveLearner.

        Keys: REL_LOOK_DOWN, REL_YAW, REL_LEAN, REL_MAR
        """
        if not self.is_ready(sid):
            return None

        thr = self._thresholds[sid]

        return {
            "REL_LOOK_DOWN": max(
                self.cfg.min_deadband_look_down,
                thr.look_down.std * self.cfg.k_look_down,
            ),
            "REL_YAW": max(
                self.cfg.min_deadband_gaze,
                thr.gaze.std * self.cfg.k_gaze,
            ),
            "REL_LEAN": max(
                self.cfg.min_deadband_lean,
                thr.lean.std * self.cfg.k_lean,
            ),
            "REL_MAR": max(
                self.cfg.min_deadband_mar,
                thr.mar.std * self.cfg.k_mar,
            ),
        }

    def calibration_progress(self, sid: str) -> Tuple[int, int]:
        """(current_samples, target_samples) for progress-bar display."""
        if sid in self._finalised:
            return self._window_frames, self._window_frames
        buf = self._buffers.get(sid)
        if buf is None:
            return 0, self._window_frames
        n = self._min_samples(buf)
        return n, self._window_frames

    def pending_sids(self) -> List[str]:
        """List of student IDs still waiting to be finalised."""
        return [sid for sid in self._buffers if sid not in self._finalised]

    def reset(self, sid: str = None):
        """Reset state for one student (or all if sid is None)."""
        if sid is None:
            self._buffers.clear()
            self._thresholds.clear()
            self._finalised.clear()
        else:
            self._buffers.pop(sid, None)
            self._thresholds.pop(sid, None)
            self._finalised.discard(sid)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_buf(self, sid: str) -> _StudentCalibBuffer:
        if sid not in self._buffers:
            self._buffers[sid] = _StudentCalibBuffer(window_start=time.time())
        return self._buffers[sid]

    @staticmethod
    def _min_samples(buf: _StudentCalibBuffer) -> int:
        """Returns the minimum sample count across all signals."""
        counts = [
            len(buf.gaze), len(buf.look_down),
            len(buf.lean), len(buf.mar), len(buf.head),
        ]
        return min(c for c in counts if c > 0) if any(c > 0 for c in counts) else 0

    def _is_stable(self, buf: _StudentCalibBuffer) -> bool:
        """
        Returns True if all signals with enough samples have
        coefficient-of-variation below cv_stable_threshold.
        """
        for signal in self.SIGNALS:
            vals = getattr(buf, signal)
            if len(vals) < self.cfg.min_samples:
                continue   # not enough data to judge this signal
            arr = np.array(vals)
            mean = float(np.mean(arr))
            std  = float(np.std(arr))
            if abs(mean) < 1e-9:
                continue   # avoid div-by-zero for near-zero signals
            cv = std / abs(mean)
            if cv > self.cfg.cv_stable_threshold:
                return False
        return True

    def _compute_signal(
        self,
        raw_vals: List[float],
        k: float,
        min_deadband: float,
        fallback: float,
        name: str,
    ) -> SignalThreshold:
        """
        From a list of raw values, produce a SignalThreshold:
          1. IQR outlier filter
          2. Median as baseline
          3. Std as spread
          4. thr_high / thr_low = baseline ± max(k*std, min_deadband)
        """
        st = SignalThreshold()

        if len(raw_vals) < self.cfg.min_samples:
            # Not enough data — use fallback
            st.baseline   = fallback
            st.std        = min_deadband
            st.thr_high   = fallback + min_deadband
            st.thr_low    = fallback - min_deadband
            st.is_fallback = True
            log.debug("[AutoThreshold] signal=%s: insufficient samples (%d) — fallback", name, len(raw_vals))
            return st

        arr = np.array(raw_vals, dtype=float)

        # ── IQR outlier removal ─────────────────────────────────────────────
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr    = q3 - q1
        lower  = q1 - self.cfg.iqr_factor * iqr
        upper  = q3 + self.cfg.iqr_factor * iqr
        clean  = arr[(arr >= lower) & (arr <= upper)]

        if len(clean) < max(self.cfg.min_samples // 2, 5):
            clean = arr   # outlier filter was too aggressive — use raw
            log.debug("[AutoThreshold] signal=%s: IQR filter too aggressive, reverting to raw", name)

        baseline = float(np.median(clean))
        std      = float(np.std(clean))
        deadband = max(k * std, min_deadband)

        st.baseline    = baseline
        st.std         = std
        st.thr_high    = baseline + deadband
        st.thr_low     = baseline - deadband
        st.is_fallback = False

        return st

    def _log_thresholds(
        self,
        sid: str,
        thr: StudentThresholds,
        buf: _StudentCalibBuffer,
    ):
        elapsed = thr.finalised_at - buf.window_start
        n_gaze  = len(buf.gaze) if buf else 0
        n_lean  = len(buf.lean) if buf else 0
        fb      = "FALLBACK" if thr.is_fallback else "personalised"

        log.info(
            "[AutoThreshold] ── %s (%s, %.1fs, %d gaze / %d lean samples) ──",
            sid, fb, elapsed, n_gaze, n_lean,
        )
        for signal in self.SIGNALS:
            st = getattr(thr, signal)
            tag = "~" if st.is_fallback else "✓"
            log.info(
                "  %s %-10s baseline=%.4f  std=%.4f  "
                "thr_low=%.4f  thr_high=%.4f",
                tag, signal, st.baseline, st.std, st.thr_low, st.thr_high,
            )


# ---------------------------------------------------------------------------
# Integration helper: auto-tick all pending students
# ---------------------------------------------------------------------------

def tick_auto_threshold(
    at: AutoThreshold,
    learner,
    student_id_map: Dict[int, str],
    current_positions: Dict[int, Any],
    face_mesh_results,
    pose_results,
    iw: int,
    ih: int,
):
    """
    Call this once per frame (during calibration phase) to:
      1. Extract signals for each visible student
      2. Feed them into AutoThreshold
      3. Finalise + apply to AdaptiveLearner when windows are complete

    Parameters
    ----------
    at               : AutoThreshold instance
    learner          : AdaptiveLearner instance
    student_id_map   : {track_id: sid}
    current_positions: {track_id: (cx, cy)}
    face_mesh_results: MediaPipe FaceMesh results object (or None)
    pose_results     : MediaPipe Pose results object (or None)
    iw, ih           : frame width / height in pixels
    """
    from adaptive_learning import AdaptiveLearner  # avoid circular at module level

    # ── Extract face signals ──────────────────────────────────────────────
    face_signals: Dict[str, Dict[str, float]] = {}

    if face_mesh_results and face_mesh_results.multi_face_landmarks:
        for f_lms in face_mesh_results.multi_face_landmarks:
            pts = f_lms.landmark
            fx  = int(pts[1].x * iw)
            fy  = int(pts[1].y * ih)

            # find nearest tracked student
            tid = AdaptiveLearner._get_student_at(fx, fy, current_positions)
            if tid is None:
                continue
            sid = student_id_map.get(tid)
            if sid is None:
                continue

            # gaze ratio
            left_iris   = pts[468].x
            right_iris  = pts[473].x
            left_outer  = pts[33].x
            left_inner  = pts[133].x
            right_outer = pts[263].x
            right_inner = pts[362].x
            l_ratio = (left_iris  - left_outer)  / (left_inner  - left_outer  + 1e-6)
            r_ratio = (right_iris - right_inner) / (right_outer - right_inner + 1e-6)
            gaze    = (l_ratio + r_ratio) / 2.0

            # look-down score (iris y relative to eyelid)
            li, ri  = pts[468], pts[473]
            ley     = (pts[159].y + pts[145].y) / 2
            rey     = (pts[386].y + pts[374].y) / 2
            ld      = ((li.y - ley) + (ri.y - rey)) / 2.0

            # MAR
            v1  = np.linalg.norm(np.array([pts[13].x, pts[13].y]) - np.array([pts[14].x, pts[14].y]))
            v2  = np.linalg.norm(np.array([pts[312].x, pts[312].y]) - np.array([pts[317].x, pts[317].y]))
            w   = np.linalg.norm(np.array([pts[61].x, pts[61].y]) - np.array([pts[291].x, pts[291].y])) + 1e-6
            mar = float(((v1 + v2) / 2.0) / w)

            # Head ratio (nose x)
            head = float(pts[1].x)

            face_signals[sid] = {
                "gaze":      float(gaze),
                "look_down": float(ld),
                "mar":       float(mar),
                "head":      float(head),
            }

    # ── Extract lean signal from pose ─────────────────────────────────────
    lean_signals: Dict[str, float] = {}

    if pose_results and pose_results.pose_landmarks:
        lms = pose_results.pose_landmarks.landmark
        tid = AdaptiveLearner._get_student_at(
            int(lms[0].x * iw), int(lms[0].y * ih), current_positions
        )
        if tid is not None:
            sid = student_id_map.get(tid)
            if sid is not None:
                lean_signals[sid] = abs(lms[11].x - lms[12].x)

    # ── Feed signals & trigger finalisation ───────────────────────────────
    all_sids = set(student_id_map.values())

    for sid in all_sids:
        if at.is_ready(sid):
            continue

        signals: Dict[str, Optional[float]] = {}

        if sid in face_signals:
            signals.update(face_signals[sid])
        if sid in lean_signals:
            signals["lean"] = lean_signals[sid]

        if signals:
            at.feed(sid, signals)

        # Check if window is complete for this student
        if at.should_finalise(sid):
            thr = at.finalise(sid)

            # Push into AdaptiveLearner baseline
            bl = learner._baselines.get(sid)
            if bl is not None:
                at.apply_to_learner(sid, bl)
            else:
                log.warning(
                    "[AutoThreshold] %s: no AdaptiveLearner baseline found — "
                    "thresholds computed but not applied",
                    sid,
                )
