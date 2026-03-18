import cv2
from ai_engine import run_ai_on_frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    out = run_ai_on_frame(frame)

    cv2.imshow("AI Proctor", out)

    if cv2.waitKey(1) == 27:
        break