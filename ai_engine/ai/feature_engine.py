class FeatureEngine:
    def build_features(self, sid, events):
        features = {
            "sid": sid,
            "look_left": 0,
            "look_right": 0,
            "look_down": 0,
            "talking": 0,
            "phone": 0,
            "leaning": 0,
            "hands_face": 0,
        }

        for e in events:
            etype = e.get("type", "")

            if etype == "LOOKING_LEFT":
                features["look_left"] += 1
            elif etype == "LOOKING_RIGHT":
                features["look_right"] += 1
            elif etype == "LOOKING_DOWN":
                features["look_down"] += 1
            elif etype == "TALKING_DETECTED":
                features["talking"] += 1
            elif etype == "PHONE_DETECTED":
                features["phone"] += 1
            elif etype == "BODY_LEANING":
                features["leaning"] += 1
            elif etype == "HANDS_ON_FACE":
                features["hands_face"] += 1

        return features