import time

class RiskEngine:
    def __init__(self):
        self.risk_scores = {}
        self.event_memory = {}
        self.COOLDOWN = 5

        self.EVENT_WEIGHTS = {
            "CALCULATOR_DETECTED": 40,
            "FACE_ABSENT": 30,
            "LOOKING_LEFT": 10,
            "BEHAVIOR_ANOMALY": 20
        }

    def should_trigger(self, student, event):
        now = time.time()

        if student not in self.event_memory:
            self.event_memory[student] = {}

        last_time = self.event_memory[student].get(event, 0)

        if now - last_time > self.COOLDOWN:
            self.event_memory[student][event] = now
            return True

        return False

    def update_risk(self, student, event):
        if student not in self.risk_scores:
            self.risk_scores[student] = 0

        if self.should_trigger(student, event):
            self.risk_scores[student] += self.EVENT_WEIGHTS.get(event, 5)

        # decay
        self.risk_scores[student] *= 0.98

        # clamp
        self.risk_scores[student] = min(self.risk_scores[student], 100)

        return self.risk_scores[student]

    def get_level(self, score):
        if score > 70:
            return "HIGH"
        elif score > 40:
            return "MEDIUM"
        else:
            return "LOW"
