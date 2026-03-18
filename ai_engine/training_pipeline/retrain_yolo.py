from ultralytics import YOLO
from config import *

def retrain():

    model = YOLO(f"{MODEL_PRODUCTION}/object_model.pt")

    model.train(
        data=DATASET_YAML,
        epochs=30,
        imgsz=640,
        batch=8,
        device="cpu"
    )

    print("✅ Retraining complete")