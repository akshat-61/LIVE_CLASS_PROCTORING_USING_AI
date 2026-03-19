# ai_engine/tracking/student_state.py

import time
import logging

log = logging.getLogger(__name__)


class StudentState:
    def __init__(self):
        self.registry = {}

    def update(self, student_id, **kwargs):
        if student_id not in self.registry:
            self.registry[student_id] = self._default_state(student_id)

        state = self.registry[student_id]

        for key, value in kwargs.items():
            state[key] = value

        state["last_updated"] = time.time()

        log.debug("State updated | sid=%s data=%s", student_id, kwargs)

    def get(self, student_id):
        return self.registry.get(student_id, None)

    def get_all(self):
        return self.registry

    def mark_absent(self, student_id):
        if student_id in self.registry:
            self.registry[student_id]["face_present"] = False
            self.registry[student_id]["last_seen"] = time.time()

    def mark_present(self, student_id):
        if student_id not in self.registry:
            self.registry[student_id] = self._default_state(student_id)

        self.registry[student_id]["face_present"] = True
        self.registry[student_id]["last_seen"] = time.time()

    def add_event(self, student_id, event):
        if student_id not in self.registry:
            self.registry[student_id] = self._default_state(student_id)

        self.registry[student_id]["events"].append(event)

    def reset(self):
        self.registry.clear()

    def _default_state(self, student_id):
        return {
            "student_id": student_id,
            "face_present": True,
            "gaze_direction": "center",
            "head_ratio": 0.5,
            "last_seen": time.time(),
            "last_updated": time.time(),
            "events": [],
            "suspicion_score": 0.0,
        }