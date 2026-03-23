import os
import time
import glob
from datetime import datetime, timedelta
from collections import defaultdict

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image as RLImage
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm

RISK_COLORS = {
    "CRITICAL": colors.HexColor("#c0392b"),
    "HIGH":     colors.HexColor("#e67e22"),
    "MEDIUM":   colors.HexColor("#f39c12"),
    "LOW":      colors.HexColor("#27ae60"),
    "CLEAR":    colors.HexColor("#2ecc71"),
}

def _risk_level(score: float) -> str:
    if score >= 80: return "CRITICAL"
    if score >= 40: return "HIGH"
    if score >= 15: return "MEDIUM"
    if score >= 5:  return "LOW"
    return "CLEAR"


def _fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "—"


def _build_styles():
    base = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle("title", parent=base["Title"],
                                fontSize=20, spaceAfter=6,
                                textColor=colors.HexColor("#1a1a2e")),
        "subtitle": ParagraphStyle("subtitle", parent=base["Normal"],
                                   fontSize=11, spaceAfter=4,
                                   textColor=colors.HexColor("#555555")),
        "h2": ParagraphStyle("h2", parent=base["Heading2"],
                              fontSize=13, spaceBefore=14, spaceAfter=4,
                              textColor=colors.HexColor("#1a1a2e")),
        "h3": ParagraphStyle("h3", parent=base["Heading3"],
                              fontSize=11, spaceBefore=8, spaceAfter=3,
                              textColor=colors.HexColor("#2c3e50")),
        "body": ParagraphStyle("body", parent=base["Normal"],
                               fontSize=9, leading=13, spaceAfter=4),
        "llm": ParagraphStyle("llm", parent=base["Normal"],
                              fontSize=9, leading=13, spaceAfter=4,
                              leftIndent=10, rightIndent=10,
                              backColor=colors.HexColor("#f8f9fa"),
                              borderColor=colors.HexColor("#dee2e6"),
                              borderPadding=6),
        "mono": ParagraphStyle("mono", parent=base["Code"],
                               fontSize=8, leading=11),
        "center": ParagraphStyle("center", parent=base["Normal"],
                                 fontSize=9, alignment=TA_CENTER),
    }
    return styles


def _header_table(exam_id: str, room_id: str, start_time: float,
                  duration_min: int, student_count: int,
                  total_events: int, styles: dict) -> Table:
    data = [
        [Paragraph("Exam ID", styles["subtitle"]),
         Paragraph(str(exam_id), styles["body"]),
         Paragraph("Room", styles["subtitle"]),
         Paragraph(str(room_id), styles["body"])],
        [Paragraph("Date", styles["subtitle"]),
         Paragraph(datetime.fromtimestamp(start_time).strftime("%d %b %Y"), styles["body"]),
         Paragraph("Start time", styles["subtitle"]),
         Paragraph(_fmt_ts(start_time), styles["body"])],
        [Paragraph("Duration", styles["subtitle"]),
         Paragraph(f"{duration_min} minutes", styles["body"]),
         Paragraph("Students", styles["subtitle"]),
         Paragraph(str(student_count), styles["body"])],
        [Paragraph("Total events", styles["subtitle"]),
         Paragraph(str(total_events), styles["body"]),
         Paragraph("Generated", styles["subtitle"]),
         Paragraph(datetime.now().strftime("%d %b %Y  %H:%M"), styles["body"])],
    ]
    t = Table(data, colWidths=[(PAGE_W - 2*MARGIN) * 0.2] * 4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f4f8")),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [colors.HexColor("#f0f4f8"), colors.HexColor("#e8edf2")]),
        ("GRID",   (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _risk_summary_table(student_scores: dict, styles: dict) -> Table:
    header = [
        Paragraph("Student", styles["subtitle"]),
        Paragraph("Score", styles["subtitle"]),
        Paragraph("Risk Level", styles["subtitle"]),
        Paragraph("Top Events", styles["subtitle"]),
        Paragraph("Alerts", styles["subtitle"]),
    ]
    rows = [header]

    sorted_students = sorted(student_scores.items(), key=lambda x: -x[1]["score"])

    for sid, data in sorted_students:
        score  = data["score"]
        level  = _risk_level(score)
        color  = RISK_COLORS.get(level, colors.black)
        events = data.get("events", [])

        event_counts = defaultdict(int)
        for e in events:
            etype = e.get("type") or e.get("event_type") or str(e)
            event_counts[etype] += 1
        top = sorted(event_counts.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{et}({c})" for et, c in top) if top else "—"

        rows.append([
            Paragraph(sid, styles["body"]),
            Paragraph(f"{score:.0f}", styles["body"]),
            Paragraph(level, ParagraphStyle("risk", parent=styles["body"],
                                            textColor=color, fontName="Helvetica-Bold")),
            Paragraph(top_str[:60], styles["body"]),
            Paragraph(str(len(events)), styles["body"]),
        ])

    col_w = (PAGE_W - 2*MARGIN)
    t = Table(rows, colWidths=[col_w*0.1, col_w*0.08, col_w*0.12, col_w*0.55, col_w*0.1])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f7f9fc")]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _timeline_table(events: list, styles: dict) -> Table:
    header = [
        Paragraph("Time", styles["subtitle"]),
        Paragraph("Student", styles["subtitle"]),
        Paragraph("Event", styles["subtitle"]),
        Paragraph("Confidence", styles["subtitle"]),
    ]
    rows = [header]

    for e in sorted(events, key=lambda x: x.get("timestamp", 0)):
        ts    = e.get("timestamp", 0)
        sid   = e.get("student_id") or e.get("trackId") or "—"
        etype = e.get("type") or e.get("event_type") or "UNKNOWN"
        conf  = e.get("confidence", 0.9)

        rows.append([
            Paragraph(_fmt_ts(ts), styles["mono"]),
            Paragraph(str(sid)[:12], styles["body"]),
            Paragraph(etype, styles["body"]),
            Paragraph(f"{conf:.0%}", styles["body"]),
        ])

    col_w = (PAGE_W - 2*MARGIN)
    t = Table(rows, colWidths=[col_w*0.12, col_w*0.12, col_w*0.60, col_w*0.12],
              repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f7f9fc")]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def generate_pdf_report(
    output_path: str,
    exam_id: str,
    room_id: str,
    start_time: float,
    duration_min: int,
    student_scores: dict,
    all_events: list,
    llm_summaries: dict = None,
    room_summary: str = None,
    evidence_dir: str = "evidence",
    seat_map_path: str = None,
) -> str:
    styles = _build_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN,  bottomMargin=MARGIN,
        title=f"Proctoring Report — {exam_id}",
        author="AI Proctoring System",
    )

    story = []

    story.append(Paragraph("AI Proctoring System", styles["subtitle"]))
    story.append(Paragraph("Exam Proctoring Report", styles["title"]))
    story.append(HRFlowable(width="100%", thickness=1.5,
                            color=colors.HexColor("#2c3e50"), spaceAfter=10))

    story.append(_header_table(
        exam_id, room_id, start_time, duration_min,
        len(student_scores), len(all_events), styles
    ))
    story.append(Spacer(1, 0.4*cm))

    if room_summary:
        story.append(Paragraph("Executive Summary", styles["h2"]))
        story.append(Paragraph(room_summary, styles["llm"]))
        story.append(Spacer(1, 0.3*cm))

    if seat_map_path and os.path.exists(seat_map_path):
        story.append(Paragraph("Seat Map", styles["h2"]))
        try:
            img = RLImage(seat_map_path, width=14*cm, height=7*cm, kind="proportional")
            story.append(img)
        except Exception:
            pass
        story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph("Student Risk Summary", styles["h2"]))
    story.append(_risk_summary_table(student_scores, styles))
    story.append(Spacer(1, 0.3*cm))

    risky = {sid: d for sid, d in student_scores.items() if d["score"] >= 15}
    if risky:
        story.append(PageBreak())
        story.append(Paragraph("Per-Student Analysis", styles["h2"]))
        story.append(Spacer(1, 0.2*cm))

        for sid, data in sorted(risky.items(), key=lambda x: -x[1]["score"]):
            score = data["score"]
            level = _risk_level(score)
            color = RISK_COLORS.get(level, colors.black)

            story.append(Paragraph(
                f'{sid}  <font color="#{_color_hex(color)}"><b>{level}</b></font>'
                f'  —  Score: {score:.0f}/100',
                styles["h3"]
            ))

            if llm_summaries and sid in llm_summaries:
                story.append(Paragraph(llm_summaries[sid], styles["llm"]))

            sid_events = [e for e in all_events
                          if (e.get("student_id") or e.get("trackId") or "") == sid]
            if sid_events:
                event_counts = defaultdict(int)
                for e in sid_events:
                    event_counts[e.get("type") or e.get("event_type") or "UNKNOWN"] += 1
                lines = [f"{et}: {c}" for et, c in
                         sorted(event_counts.items(), key=lambda x: -x[1])]
                story.append(Paragraph("Events: " + "  |  ".join(lines), styles["body"]))

            evidence_photos = _find_evidence_photos(evidence_dir, sid)
            if evidence_photos:
                photo_row = []
                for photo_path in evidence_photos[:4]:
                    try:
                        img = RLImage(photo_path, width=3.5*cm, height=2.5*cm,
                                      kind="proportional")
                        photo_row.append(img)
                    except Exception:
                        pass
                if photo_row:
                    while len(photo_row) < 4:
                        photo_row.append("")
                    t = Table([photo_row], colWidths=[3.8*cm]*4)
                    t.setStyle(TableStyle([
                        ("ALIGN",  (0,0), (-1,-1), "CENTER"),
                        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                        ("BOX",    (0,0), (-1,-1), 0.3, colors.HexColor("#cccccc")),
                    ]))
                    story.append(t)

            story.append(Spacer(1, 0.3*cm))

    story.append(PageBreak())
    story.append(Paragraph("Full Event Timeline", styles["h2"]))
    if all_events:
        story.append(_timeline_table(all_events, styles))
    else:
        story.append(Paragraph("No events recorded.", styles["body"]))

    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor("#cccccc")))
    story.append(Paragraph(
        f"Generated by AI Proctoring System  —  {datetime.now().strftime('%d %b %Y %H:%M:%S')}",
        styles["center"]
    ))

    doc.build(story)
    return output_path


def _color_hex(color) -> str:
    try:
        r = int(color.red * 255)
        g = int(color.green * 255)
        b = int(color.blue * 255)
        return f"{r:02x}{g:02x}{b:02x}"
    except Exception:
        return "000000"


def _find_evidence_photos(evidence_dir: str, sid: str) -> list:
    photos = []
    if not os.path.exists(evidence_dir):
        return photos
    for root, dirs, files in os.walk(evidence_dir):
        for f in files:
            if sid in f and f.lower().endswith((".jpg", ".jpeg", ".png")):
                photos.append(os.path.join(root, f))
    return sorted(photos)[:4]


def generate_report_from_session(
    output_dir: str,
    exam_id: str,
    room_id: str,
    event_manager,
    score_engine,
    student_state,
    llm_analyzer=None,
    evidence_dir: str = "evidence",
    seat_map_path: str = None,
    exam_duration_min: int = 60,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    all_events = event_manager.get_all_events() if hasattr(event_manager, "get_all_events") else []
    all_scores = score_engine.get_all_scores() if hasattr(score_engine, "get_all_scores") else {}
    all_states = student_state.get_all() if hasattr(student_state, "get_all") else {}

    student_scores = {}
    for sid, score in all_scores.items():
        sid_events = [e for e in all_events
                      if (e.get("student_id") or e.get("trackId") or "") == sid
                      or str(e).startswith(sid)]
        student_scores[sid] = {
            "score":    score,
            "events":   sid_events,
            "seat_pos": all_states.get(sid, {}).get("position"),
        }

    start_time = time.time() - exam_duration_min * 60

    llm_summaries = {}
    room_summary  = None
    if llm_analyzer is not None:
        risky = {sid: d for sid, d in student_scores.items() if d["score"] >= 15}
        for sid, data in risky.items():
            llm_summaries[sid] = llm_analyzer.analyse_student(
                sid, data["events"], data["score"], data.get("seat_pos")
            )
        students_for_room = [
            {"sid": sid, "score": d["score"]} for sid, d in student_scores.items()
        ]
        room_summary = llm_analyzer.analyse_room(
            students_for_room, len(all_events), exam_duration_min
        )

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"report_{exam_id}_{timestamp}.pdf")

    generate_pdf_report(
        output_path    = output_path,
        exam_id        = exam_id,
        room_id        = room_id,
        start_time     = start_time,
        duration_min   = exam_duration_min,
        student_scores = student_scores,
        all_events     = all_events,
        llm_summaries  = llm_summaries,
        room_summary   = room_summary,
        evidence_dir   = evidence_dir,
        seat_map_path  = seat_map_path,
    )

    return output_path
