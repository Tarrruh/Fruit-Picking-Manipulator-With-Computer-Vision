import cv2
import time
import os
from ultralytics import YOLO

base_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(base_path, "..", ".."))

model_path = os.path.join(project_root, "runs", "classify", "train", "weights", "best.pt")

print(f"Loading model from: {model_path}")

model = YOLO(model_path)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, verbose=False)[0]
    
    if results.probs is not None:
        cls_id = results.probs.top1
        conf = results.probs.top1conf.item()
        label = model.names[cls_id]

        color = (0, 255, 0) if "ripe" in label.lower() and "unripe" not in label.lower() else (0, 0, 255)
        cv2.putText(frame, f"{label} {conf:.2%}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()