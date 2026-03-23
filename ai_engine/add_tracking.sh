#!/bin/bash

echo "🚀 Adding DeepSORT Tracking to AI Proctoring System..."

# =========================
# 1. INSTALL DEPENDENCY
# =========================
echo "📦 Installing DeepSORT..."
pip install deep_sort_realtime

# =========================
# 2. CREATE TRACKING MODULE
# =========================
echo "🧠 Creating tracking module..."

mkdir -p tracking

cat <<EOL > tracking/tracker.py
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self):
        self.tracker = DeepSort(
            max_age=40,
            n_init=2,
            max_cosine_distance=0.2
        )

    def update(self, detections, frame):
        """
        detections: [[x1, y1, x2, y2, confidence]]
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            l, t, r, b = track.to_ltrb()
            track_id = track.track_id

            results.append({
                "id": f"S{track_id:03}",
                "bbox": [int(l), int(t), int(r), int(b)]
            })

        return results
EOL

echo "✅ Tracker created"

# =========================
# 3. CREATE UTILS FOR MAPPING
# =========================
echo "🔗 Creating bbox mapping utils..."

mkdir -p utils

cat <<EOL > utils/bbox_utils.py
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if boxAArea + boxBArea - interArea == 0:
        return 0

    return interArea / float(boxAArea + boxBArea - interArea)


def assign_object_to_person(persons, objects):
    mapping = {}

    for person in persons:
        best_iou = 0
        assigned_obj = None

        for obj in objects:
            score = iou(person["bbox"], obj["bbox"])
            if score > best_iou:
                best_iou = score
                assigned_obj = obj

        if best_iou > 0.3:
            mapping[person["id"]] = assigned_obj

    return mapping
EOL

echo "✅ BBox utils created"

# =========================
# 4. CREATE SAMPLE INTEGRATION PATCH
# =========================
echo "🧩 Creating integration guide..."

cat <<EOL > tracking/INTEGRATION_GUIDE.txt

===============================
TRACKING INTEGRATION STEPS
===============================

1. Import tracker:

from tracking.tracker import Tracker

2. Initialize (once):

tracker = Tracker()

3. Convert YOLO detections:

detections = []
for box in yolo_boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    conf = box.conf[0]
    detections.append([x1, y1, x2, y2, conf])

4. Update tracker:

tracked = tracker.update(detections, frame)

5. Use tracked output:

for obj in tracked:
    student_id = obj["id"]
    bbox = obj["bbox"]

6. Remove old ID generation logic ❌

7. Map objects:

from utils.bbox_utils import assign_object_to_person

mapping = assign_object_to_person(tracked, detected_objects)

===============================

EOL

echo "✅ Integration guide created"

# =========================
# 5. CLEAN DUPLICATE ID LOGIC WARNING
# =========================
echo ""
echo "⚠️ IMPORTANT:"
echo "-----------------------------------"
echo "1. REMOVE any random ID generation"
echo "2. REMOVE per-frame student assignment"
echo "3. ONLY use tracker IDs"
echo "-----------------------------------"

# =========================
# DONE
# =========================
echo ""
echo "🎉 TRACKING SETUP COMPLETE!"
echo ""
echo "👉 NEXT:"
echo "Integrate tracker in ai_engine.py using guide"
echo "Then run: python process_video.py"
