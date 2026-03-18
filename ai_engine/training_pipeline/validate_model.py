from ultralytics import YOLO

def validate(model_path):

    model = YOLO(model_path)

    metrics = model.val()

    print("📊 Validation Results:")
    print(metrics)