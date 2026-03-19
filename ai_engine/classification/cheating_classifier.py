# ai_engine/classification/cheating_classifier.py

import time
import logging

log = logging.getLogger(__name__)


class CheatingClassifier:
    def __init__(self, window_seconds=60):
        self.window_seconds = window_seconds

        # Event weights (tunable)
        self.weights = {
            "PHONE_DETECTED": 3.0,
            "LOOKING_DOWN": 1.5,
            "LOOKING_LEFT": 1.0,
            "LOOKING_RIGHT": 1.0,
            "TALKING_DETECTED": 2.0,
            "WHISPERING_DETECTED": 3.5,
            "PASSING_OBJECT": 4.0,
            "HANDS_ON_FACE": 1.2,
            "WRITING_ON_PALM": 3.0,
            "BODY_LEANING": 1.3,
            "CALCULATOR_DETECTED": 2.5,
            "CALCULATOR_PERSISTENT": 4.5,
            "ID_NOT_VISIBLE": 1.5,
            "SEAT_VACATED": 5.0,
            "SUSPICIOUS_GESTURE": 2.0,
        }

        # Thresholds
        self.suspicious_threshold = 5.0
        self.cheating_threshold = 10.0

    def classify(self, student_id, events):
        """
        events: list of event dicts from event_manager
        """

        now = time.time()
        score = 0.0

        for e in events:
            event_type = e["event_type"]
            timestamp = e["timestamp"]

            # Skip old events
            age = now - timestamp
            if age > self.window_seconds:
                continue

            # Decay (recent events matter more)
            decay = 1 - (age / self.window_seconds)

            weight = self.weights.get(event_type, 0.5)
            score += weight * decay

        label = self._get_label(score)

        log.debug(
            "Classifier | sid=%s score=%.2f label=%s",
            student_id,
            score,
            label
        )

        return {
            "student_id": student_id,
            "score": round(score, 2),
            "label": label
        }

    def _get_label(self, score):
        if score >= self.cheating_threshold:
            return "CHEATING"
        elif score >= self.suspicious_threshold:
            return "SUSPICIOUS"
        return "NORMAL"

    def classify_all(self, event_manager):
        results = {}

        all_events = event_manager.get_all_events()

        for sid, events in all_events.items():
            results[sid] = self.classify(sid, events)

        return results