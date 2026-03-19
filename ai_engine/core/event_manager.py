# ai_engine/core/event_manager.py

import time
from collections import defaultdict, deque
import logging

log = logging.getLogger(__name__)


class EventManager:
    def __init__(self, max_events_per_student=500):
        self.events = defaultdict(lambda: deque(maxlen=max_events_per_student))

    def emit(self, student_id, event_type, confidence=1.0, metadata=None):
        event = {
            "student_id": student_id,
            "event_type": event_type,
            "timestamp": time.time(),
            "confidence": confidence,
            "metadata": metadata or {}
        }

        self.events[student_id].append(event)

        log.debug(
            "Event emitted | sid=%s type=%s conf=%.2f",
            student_id,
            event_type,
            confidence
        )

        return event

    def get_events(self, student_id):
        return list(self.events.get(student_id, []))

    def get_recent_events(self, student_id, seconds=60):
        now = time.time()
        return [
            e for e in self.events.get(student_id, [])
            if now - e["timestamp"] <= seconds
        ]

    def get_all_events(self):
        return {sid: list(events) for sid, events in self.events.items()}

    def clear_student(self, student_id):
        self.events.pop(student_id, None)

    def reset(self):
        self.events.clear()