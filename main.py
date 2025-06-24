import cv2
import json
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict

# ========== CONFIG ==========
VIDEO_PATH = '20sec.mp4'
FRAME_INTERVAL = 5
ANNOTATED_DIR = 'annotated_frames'
OUTPUT_JSON = 'detection_results.json'
OUTPUT_IMG = 'object_frequency1.png'
OUTPUT_VIDEO = 'output_video.avi'
MODEL_PATH = 'yolov8n.pt'  # Uses pretrained YOLOv5s model
# ============================

# Create directory for annotated frames
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# Load model
print("[INFO] Loading YOLOv5 model...")
model = YOLO(MODEL_PATH)

# Load video
print("[INFO] Reading video...")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_id = 0
frames = []
detections_json = []
class_counts = defaultdict(int)
max_class_diversity = 0
most_diverse_frame = None

# Process every 5th frame
print("[INFO] Processing frames...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % FRAME_INTERVAL == 0:
        results = model(frame, verbose=False)[0]
        frame_classes = set()

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls.item())
            label = model.names[cls_id]
            conf = float(box.conf.item())
            bbox = list(map(float, box.xyxy[0]))

            detections.append({
                "label": label,
                "bbox": bbox,
                "confidence": conf
            })

            class_counts[label] += 1
            frame_classes.add(label)

        detections_json.append({
            "frame_id": frame_id,
            "class_diversity": len(frame_classes),
            "detections": detections
        })

        if len(frame_classes) > max_class_diversity:
            max_class_diversity = len(frame_classes)
            most_diverse_frame = frame_id

        # Save annotated frame
        annotated = results.plot()
        cv2.imwrite(f"{ANNOTATED_DIR}/frame_{frame_id}.jpg", annotated)

        # Store for output video
        frames.append(annotated)

    frame_id += 1

cap.release()

# Save output JSON
with open(OUTPUT_JSON, 'w') as f:
    json.dump(detections_json, f, indent=4)

print(f"[INFO] Detection results saved to {OUTPUT_JSON}")
print(f"[INFO] Frame with most class diversity: {most_diverse_frame} with {max_class_diversity} classes")

# Plot object frequency
print("[INFO] Generating bar chart...")
plt.figure(figsize=(10, 6))
plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
plt.xticks(rotation=45)
plt.xlabel("Object Classes")
plt.ylabel("Frequency")
plt.title("Object Frequency Across Video")
plt.tight_layout()
plt.savefig(OUTPUT_IMG)
plt.show()
print(f"[INFO] Bar chart saved to {OUTPUT_IMG}")

# Optional: Save output video
print("[INFO] Creating annotated output video...")
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'XVID'), fps // FRAME_INTERVAL, (frame_width, frame_height))
for frame in frames:
    out.write(frame)
out.release()
print(f"[INFO] Output video saved to {OUTPUT_VIDEO}")
