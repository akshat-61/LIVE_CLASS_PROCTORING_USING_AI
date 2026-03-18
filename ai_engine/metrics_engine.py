from collections import deque

class MetricsEngine:
    def __init__(self, window_size=120):
        """
        window_size = number of frames to analyze
        120 frames ≈ ~4–5 seconds depending on FPS
        """
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, features):
        """
        Add new frame features to buffer.
        FIX 6 (v12): features dict may now include 'student_hands_detected'
        (boolean — hands belonging to known students only, invigilator excluded).
        Falls back to 'hand_detected' if not present for backward compatibility.
        """
        self.buffer.append(features)

        if len(self.buffer) < self.window_size:
            return None

        return self.compute_metrics()

    def compute_metrics(self):
        """
        Convert feature buffer into behavior metrics.
        FIX 6 (v12): hand_activity_rate uses student-only hand detection
        to prevent invigilator arm movements from inflating the metric.
        """
        total = len(self.buffer)

        lookaway = sum(1 for f in self.buffer if f["look_away"])
        # FIX 6: prefer student-only hand count; fall back to legacy key
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