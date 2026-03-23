# tracking/student_state.py
import time
from collections import defaultdict

class StudentState:
    def __init__(self):
        self._states = {}

    def update(self, sid, face_present=True, position=None, gaze=None):
        if sid not in self._states:
            self._states[sid] = {
                "face_present": face_present,
                "position": position,
                "gaze": gaze,
                "last_seen": time.time(),
                "events": []
            }
        else:
            s = self._states[sid]
            s["face_present"] = face_present
            if position is not None:
                s["position"] = position
            if gaze is not None:
                s["gaze"] = gaze
            s["last_seen"] = time.time()

    def add_event(self, sid, event):
        if sid not in self._states:
            self._states[sid] = {"events": [], "last_seen": time.time()}
        self._states[sid].setdefault("events", []).append(event)

    def get(self, sid):
        return self._states.get(sid, {})

    def get_all(self):
        return dict(self._states)

    def reset(self):
        self._states.clear()
