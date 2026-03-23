# model_loader.py  (root level — wraps core/model_loader if needed)
import os
from ultralytics import YOLO

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "models", "production"
)

def load_models():
    print("[AI] Loading models...")
    person_model  = YOLO(os.path.join(MODEL_DIR, "person_model.pt"))
    gesture_model = YOLO(os.path.join(MODEL_DIR, "gesture_model.pt"))
    object_model  = YOLO(os.path.join(MODEL_DIR, "object_model.pt"))
    idcard_model  = YOLO(os.path.join(MODEL_DIR, "id_model.pt"))
    print("[AI] All models loaded.")
    return person_model, gesture_model, object_model, idcard_model
