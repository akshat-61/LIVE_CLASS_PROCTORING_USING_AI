# ai_engine/output/report_generator.py

import os
import time
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


class ReportGenerator:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()

    def generate(self, event_manager, classifier, evidence_manager):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_dir, f"report_{timestamp}.pdf")

        doc = SimpleDocTemplate(path, pagesize=A4)
        elements = []

        # ---------------- HEADER ----------------
        elements.append(Paragraph("AI PROCTORING REPORT", self.styles["Title"]))
        elements.append(Spacer(1, 12))

        results = classifier.classify_all(event_manager)

        # ---------------- SUMMARY ----------------
        normal = sum(1 for r in results.values() if r["label"] == "NORMAL")
        suspicious = sum(1 for r in results.values() if r["label"] == "SUSPICIOUS")
        cheating = sum(1 for r in results.values() if r["label"] == "CHEATING")

        summary_data = [
            ["Metric", "Value"],
            ["Total Students", str(len(results))],
            ["Normal", str(normal)],
            ["Suspicious", str(suspicious)],
            ["Cheating", str(cheating)],
        ]

        table = Table(summary_data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 20))

        # ---------------- STUDENT DETAILS ----------------
        all_events = event_manager.get_all_events()
        all_evidence = evidence_manager.get_all_evidence()

        for sid, result in results.items():
            elements.append(Paragraph(f"Student: {sid}", self.styles["Heading2"]))
            elements.append(Paragraph(f"Label: {result['label']} | Score: {result['score']}", self.styles["Normal"]))
            elements.append(Spacer(1, 10))

            # Events
            elements.append(Paragraph("Events:", self.styles["Heading3"]))
            events = all_events.get(sid, [])

            for e in events[-10:]:  # last 10 events
                ts = time.strftime("%H:%M:%S", time.localtime(e["timestamp"]))
                elements.append(Paragraph(f"[{ts}] {e['event_type']}", self.styles["Normal"]))

            elements.append(Spacer(1, 10))

            # Evidence Images
            elements.append(Paragraph("Evidence:", self.styles["Heading3"]))
            evidence_list = all_evidence.get(sid, [])

            for ev in evidence_list[:3]:  # max 3 images
                if os.path.exists(ev["path"]):
                    try:
                        img = Image(ev["path"], width=200, height=150)
                        elements.append(img)
                        elements.append(Spacer(1, 10))
                    except:
                        pass

            elements.append(Spacer(1, 20))

        doc.build(elements)
        return path