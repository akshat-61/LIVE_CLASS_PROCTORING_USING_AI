class AnomalyDetector:

    def __init__(self):
        self.thresholds = {
            "lookaway_rate": 0.45,
            "hand_activity_rate": 0.60,
            "gesture_rate": 0.20,
            "phone_presence_rate": 0.08,
            "multiple_people_rate": 0.05
        }

    def detect(self, metrics):

        anomalies = []

        for metric, value in metrics.items():

            threshold = self.thresholds.get(metric)

            if threshold is None:
                continue

            if value > threshold:

                ratio = value / threshold

                if ratio > 2:
                    severity = "HIGH"
                elif ratio > 1.5:
                    severity = "MEDIUM"
                else:
                    severity = "LOW"

                anomalies.append({
                    "metric": metric,
                    "value": value,
                    "threshold": threshold,
                    "severity": severity
                })

        return anomalies