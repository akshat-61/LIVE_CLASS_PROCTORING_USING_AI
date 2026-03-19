# ai_engine/output/report_generator.py

import os
import time
import logging

log = logging.getLogger(__name__)


class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(
        self,
        event_manager,
        student_state,
        classifier,
        evidence_manager,
        filename=None
    ):
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        if filename is None:
            filename = f"report_{timestamp}.txt"

        path = os.path.join(self.output_dir, filename)

        all_events = event_manager.get_all_events()
        all_states = student_state.get_all()
        all_evidence = evidence_manager.get_all_evidence()

        results = classifier.classify_all(event_manager)

        with open(path, "w") as f:
            self._write_header(f)
            self._write_summary(f, results)
            self._write_student_details(f, all_events, results, all_evidence)

        log.info("Report generated → %s", path)

        return path

    def _write_header(self, f):
        f.write("=" * 60 + "\n")
        f.write("AI PROCTORING REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated At: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def _write_summary(self, f, results):
        f.write("SUMMARY\n")
        f.write("-" * 60 + "\n")

        normal = sum(1 for r in results.values() if r["label"] == "NORMAL")
        suspicious = sum(1 for r in results.values() if r["label"] == "SUSPICIOUS")
        cheating = sum(1 for r in results.values() if r["label"] == "CHEATING")

        f.write(f"Total Students: {len(results)}\n")
        f.write(f"Normal: {normal}\n")
        f.write(f"Suspicious: {suspicious}\n")
        f.write(f"Cheating: {cheating}\n\n")

    def _write_student_details(self, f, all_events, results, all_evidence):
        f.write("STUDENT DETAILS\n")
        f.write("=" * 60 + "\n\n")

        for sid, result in results.items():
            f.write(f"Student: {sid}\n")
            f.write(f"Final Label: {result['label']}\n")
            f.write(f"Score: {result['score']}\n")

            f.write("\nEvents:\n")
            events = all_events.get(sid, [])

            for e in events:
                ts = time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))
                f.write(f"  - [{ts}] {e['event_type']} (conf={e['confidence']})\n")

            f.write("\nEvidence:\n")
            evidence = all_evidence.get(sid, [])

            for ev in evidence:
                f.write(f"  - {ev['event_type']} → {ev['path']}\n")

            f.write("\n" + "-" * 60 + "\n\n")