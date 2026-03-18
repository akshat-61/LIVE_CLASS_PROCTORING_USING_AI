class FeatureCollector:
    def __init__(self):
        pass

    def collect(
        self,
        head_pose,
        hands_detected,
        gesture,
        objects_detected,
        person_count
    ):
        features = {}

        if head_pose in ["left", "right"]:
            features["look_away"] = True
        else:
            features["look_away"] = False

        features["hand_detected"] = hands_detected

        features["gesture"] = gesture

        features["phone_detected"] = "cell phone" in objects_detected

        features["multiple_people"] = person_count > 1

        return features