from collections import deque

class MetricsEngine:
    def __init__(self, window_size=120):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, features):
        self.buffer.append(features)

        if len(self.buffer) < self.window_size:
            return None

        return self.compute_metrics()

    def compute_metrics(self):
        total = len(self.buffer)

        lookaway = sum(1 for f in self.buffer if f["look_away"])
        hands    = sum(1 for f in self.buffer
                       if f.get("student_hands_detected", f.get("hand_detected", False)))
        gesture  = sum(1 for f in self.buffer if f["gesture"] != "none")
        phone    = sum(1 for f in self.buffer if f["phone_detected"])
        multi    = sum(1 for f in self.buffer if f["multiple_people"])

        metrics = {
            "lookaway_rate":       lookaway / total,
            "hand_activity_rate":  hands    / total,
            "gesture_rate":        gesture  / total,
            "phone_presence_rate": phone    / total,
            "multiple_people_rate":multi    / total,
        }

        return metrics