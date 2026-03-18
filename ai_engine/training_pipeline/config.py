import os

BASE_DIR = "/home/tx0978/Documents/classroom-proctoring/ai-engine"

CAPTURED_FRAMES = os.path.join(BASE_DIR, "captured_frames")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

TRAIN_IMAGES = os.path.join(DATASET_DIR, "images/train")
TRAIN_LABELS = os.path.join(DATASET_DIR, "labels/train")

VAL_IMAGES = os.path.join(DATASET_DIR, "images/val")
VAL_LABELS = os.path.join(DATASET_DIR, "labels/val")

MODEL_PRODUCTION = os.path.join(BASE_DIR, "models/production")
MODEL_CANDIDATE = os.path.join(BASE_DIR, "models/candidate")

DATASET_YAML = os.path.join(BASE_DIR, "dataset_yaml/proctor_dataset.yaml")