from ultralytics import YOLO

MODEL_DIR = "models/production"

def load_models():
    print("[AI] Loading models...")

    person_model = YOLO(f"{MODEL_DIR}/person_model.pt")
    gesture_model = YOLO(f"{MODEL_DIR}/gesture_model.pt")
    object_model = YOLO(f"{MODEL_DIR}/object_model.pt")
    idcard_model = YOLO(f"{MODEL_DIR}/id_model.pt")

    print("[AI] All models loaded successfully.")

    return person_model, gesture_model, object_model, idcard_model