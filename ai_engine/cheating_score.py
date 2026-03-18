import time

class CheatingScore:

    def __init__(self):

        self.scores = {}

        # Alert cooldown times (seconds)
        self.alert_cooldown = {
            "HANDS_ON_FACE": 4,
            "BODY_LEANING": 5,
            "LOOKING_LEFT": 3,
            "LOOKING_RIGHT": 3,
            "PHONE_DETECTED": 6,
            "SUSPICIOUS_GESTURE": 5,
            "BEHAVIOR_ANOMALY": 5
        }

        # Track last time alert fired
        self.last_alert_time = {}

        # Event weights
        self.weights = {
            "PHONE_DETECTED": 40,
            "PASSING_OBJECT": 60,
            "WHISPERING_DETECTED": 30,
            "UNAUTHORIZED_OBJECT": 50,
            "LOOKING_AT_PHONE": 35,
            "BEHAVIOR_ANOMALY": 20
        }

    def can_trigger(self, student_id, event):

        key = f"{student_id}_{event}"
        now = time.time()

        cooldown = self.alert_cooldown.get(event, 3)

        if key not in self.last_alert_time:
            self.last_alert_time[key] = now
            return True

        if now - self.last_alert_time[key] > cooldown:
            self.last_alert_time[key] = now
            return True

        return False


    def add_event(self, student_id, event):

        if student_id not in self.scores:
            self.scores[student_id] = 0

        # Check cooldown
        if not self.can_trigger(student_id, event):
            return self.scores[student_id]

        weight = self.weights.get(event, 5)

        self.scores[student_id] += weight

        return self.scores[student_id]


    def decay_scores(self):

        for sid in self.scores:
            self.scores[sid] *= 0.98


    def get_score(self, student_id):
        return self.scores.get(student_id, 0)

    def get_all_scores(self):
        """FIX 9 (v12): return all student scores sorted by risk, highest first."""
        return dict(sorted(self.scores.items(), key=lambda x: -x[1]))