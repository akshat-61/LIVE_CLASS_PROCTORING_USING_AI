from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/train4/weights/best.pt")

# Input video
video_path = "Student_Exam_Cheating_CCTV_Video1.mp4"

# Run prediction
results = model.predict(
    source=video_path,
    conf=0.30,        # detection confidence
    iou=0.5,          # NMS overlap threshold
    imgsz=832,        # same size used in training
    save=True,        # save output video
    show=True,        # display live detection
    stream=False,
    max_det=100
)

print("Detection completed.")