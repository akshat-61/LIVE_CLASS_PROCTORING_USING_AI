import cv2
from ai_engine import run_ai_on_frame

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam. Try index 1 or 2.")
    exit(1)

print("[INFO] Press ESC to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame not received.")
        break

    out = run_ai_on_frame(frame)
    cv2.imshow("AI Proctor", out)

    if cv2.waitKey(1) == 27:   # ESC
        break

cap.release()
cv2.destroyAllWindows()
