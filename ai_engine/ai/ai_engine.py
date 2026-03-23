class AIEngine:

    def __init__(self):
        self.low_threshold = 40
        self.high_threshold = 70

    def rule_score(self, f):
        score = 0
        reasons = []

        if f["look_left"] >= 3:
            score += 20
            reasons.append("Repeated looking left")

        if f["look_right"] >= 3:
            score += 20
            reasons.append("Repeated looking right")

        if f["look_down"] >= 3:
            score += 15
            reasons.append("Looking down frequently")

        if f["talking"] >= 2:
            score += 25
            reasons.append("Talking detected")

        if f["phone"] >= 1:
            score += 40
            reasons.append("Phone detected")

        if f["leaning"] >= 2:
            score += 15
            reasons.append("Leaning behavior")

        if f["hands_face"] >= 2:
            score += 10
            reasons.append("Hands on face")

        # 🔥 Compound logic
        if f["look_left"] >= 2 and f["talking"] >= 1:
            score += 20
            reasons.append("Possible copying pattern")

        if f["look_down"] >= 2 and f["hands_face"] >= 1:
            score += 15
            reasons.append("Suspicious desk interaction")

        return score, reasons

    def analyze(self, features):
        score, reasons = self.rule_score(features)

        label = "LOW_RISK"
        if score > 80:
            label = "HIGH_CHEATING"
        elif score > 50:
            label = "MEDIUM_RISK"

        return {
            "sid": features["sid"],
            "score": score,
            "label": label,
            "reasons": reasons
        }