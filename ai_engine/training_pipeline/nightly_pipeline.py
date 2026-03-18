from dataset_builder import build_dataset
from retrain_yolo import retrain
from validate_model import validate
from deploy_model import deploy

def run_pipeline():

    print("📦 Building dataset...")
    build_dataset()

    print("🧠 Retraining YOLO...")
    retrain()

    print("📊 Validating model...")
    validate("runs/detect/train/weights/best.pt")

    print("🚀 Deploying model...")
    deploy("runs/detect/train/weights/best.pt")

    print("✅ Nightly pipeline finished")

if __name__ == "__main__":
    run_pipeline()