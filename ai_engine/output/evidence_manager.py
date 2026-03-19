# ai_engine/output/evidence_manager.py

import os
import cv2
import time
import logging
from collections import defaultdict

log = logging.getLogger(__name__)


class EvidenceManager:
    def __init__(self, base_dir="evidence_output", max_per_student=10):
        self.base_dir = base_dir
        self.max_per_student = max_per_student
        self.evidence = defaultdict(list)

        os.makedirs(self.base_dir, exist_ok=True)

    def capture(self, frame, student_id, event_type):
        """
        Save frame as evidence when suspicious event occurs
        """
        if len(self.evidence[student_id]) >= self.max_per_student:
            return None

        timestamp = int(time.time() * 1000)

        filename = f"{student_id}_{event_type}_{timestamp}.jpg"
        path = os.path.join(self.base_dir, filename)

        try:
            cv2.imwrite(path, frame)
            self.evidence[student_id].append({
                "event_type": event_type,
                "path": path,
                "timestamp": timestamp
            })

            log.info(
                "Evidence captured | sid=%s type=%s",
                student_id,
                event_type
            )

            return path

        except Exception as e:
            log.error("Failed to save evidence: %s", e)
            return None

    def get_evidence(self, student_id):
        return self.evidence.get(student_id, [])

    def get_all_evidence(self):
        return dict(self.evidence)

    def reset(self):
        self.evidence.clear()