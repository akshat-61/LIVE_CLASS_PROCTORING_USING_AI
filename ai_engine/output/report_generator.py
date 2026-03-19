"""
report_generator.py  —  Comprehensive end-of-session PDF report.

Six sections:
  1. Session metadata  (exam ID, room, FPS, duration, processed frames)
  2. Alert summary     (count table + horizontal bar chart per alert type)
  3. Timeline          (full event list with timestamps and student IDs)
  4. Per-student profile (score, risk level, event breakdown, evidence images)
  5. Classifier output (NORMAL / SUSPICIOUS / CHEATING verdict per student)
  6. Room analytics    (anomaly rates, activity summary)

Usage:
    from output.report_generator import generate_report

    generate_report(
        output_path   = "evidence_output/report.pdf",
        session_meta  = {...},
        alert_summary = {"SEAT_VACATED": 3, ...},
        timeline      = [{"video_time": "0:00:31", "event": ..., "student_id": ...}, ...],
        student_scores= [("S018", 78.2, "HIGH"), ...],          # from cheating_score
        classifications= {"S018": Classification(...), ...},     # from cheating_classifier
        room_metrics  = {"hand_activity_rate": 0.61, ...},
        evidence_dir  = "evidence",
    )
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── ReportLab imports ─────────────────────────────────────────────────────────
try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.platypus import (
        Flowable, HRFlowable, Image, PageBreak,
        Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )
    _RL_OK = True
except ImportError:
    _RL_OK = False
    log.warning("ReportLab not installed — PDF generation unavailable. "
                "Install with: pip install reportlab")

# ── Colour palette ────────────────────────────────────────────────────────────
_RED    = colors.HexColor("#E24B4A")
_ORANGE = colors.HexColor("#EF9F27")
_AMBER  = colors.HexColor("#BA7517")
_GREEN  = colors.HexColor("#3B6D11")
_BLUE   = colors.HexColor("#185FA5")
_TEAL   = colors.HexColor("#0F6E56")
_GRAY   = colors.HexColor("#5F5E5A")
_LGRAY  = colors.HexColor("#F1EFE8")
_WHITE  = colors.white
_BLACK  = colors.black

_RISK_COLORS = {
    "CRITICAL": _RED,
    "HIGH":     _ORANGE,
    "MEDIUM":   _AMBER,
    "LOW":      _GREEN,
}
_CLASS_COLORS = {
    "CHEATING":   _RED,
    "SUSPICIOUS": _ORANGE,
    "NORMAL":     _GREEN,
}

# ── Config ────────────────────────────────────────────────────────────────────
try:
    from config.config import cfg as _cfg
    _MAX_EVIDENCE = int(_cfg.report.max_evidence_images)
    _INCLUDE_EV   = bool(_cfg.report.include_evidence)
except Exception:
    _MAX_EVIDENCE, _INCLUDE_EV = 3, True


# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    output_path:    str | Path,
    session_meta:   Dict[str, Any],
    alert_summary:  Dict[str, int],
    timeline:       List[Dict[str, Any]],
    student_scores: List[Tuple[str, float, str]],    # (sid, score, risk_level)
    classifications: Optional[Dict[str, Any]] = None,
    room_metrics:   Optional[Dict[str, float]] = None,
    evidence_dir:   str | Path = "evidence",
    logo_path:      str | Path | None = None,
) -> Path:
    """
    Build the PDF and write it to output_path.
    Returns the Path of the created file.
    """
    if not _RL_OK:
        log.error("ReportLab not available — writing plain-text fallback")
        txt_path = Path(str(output_path).replace(".pdf", ".txt"))
        _plain_text_fallback(txt_path, session_meta, alert_summary, timeline, student_scores)
        return txt_path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title=session_meta.get("title", "AI Proctoring Report"),
        author="ALTIMA Security Technologies",
    )

    styles = _build_styles()
    story  = []

    _section_header(story, styles, "AI Proctoring — Exam Session Report", cover=True)
    _section_1_metadata(story, styles, session_meta, logo_path)
    story.append(PageBreak())

    _section_2_alert_summary(story, styles, alert_summary)
    story.append(Spacer(1, 0.5*cm))

    _section_3_timeline(story, styles, timeline)
    story.append(PageBreak())

    _section_4_per_student(story, styles, student_scores, evidence_dir, classifications)
    story.append(PageBreak())

    if classifications:
        _section_5_classifier(story, styles, classifications)
        story.append(Spacer(1, 0.5*cm))

    if room_metrics:
        _section_6_analytics(story, styles, room_metrics, alert_summary)

    doc.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    log.info("PDF report saved → %s", output_path)
    return output_path


# ── Section builders ──────────────────────────────────────────────────────────

def _section_header(story, styles, title: str, cover: bool = False) -> None:
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(title, styles["title"] if cover else styles["section"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=_GRAY))
    story.append(Spacer(1, 0.3*cm))


def _section_1_metadata(story, styles, meta: Dict[str, Any],
                          logo_path: Optional[Path]) -> None:
    _section_header(story, styles, "1. Session information")

    if logo_path and Path(logo_path).exists():
        story.append(Image(str(logo_path), width=4*cm, height=2*cm))
        story.append(Spacer(1, 0.3*cm))

    rows = [
        ["Field", "Value"],
        ["Exam ID",          meta.get("exam_id",     "—")],
        ["Room",             meta.get("room_id",      "—")],
        ["Video file",       meta.get("video_path",   "—")],
        ["Processed at",     meta.get("processed_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))],
        ["Duration",         meta.get("duration",     "—")],
        ["FPS",              str(meta.get("fps",  "—"))],
        ["Total frames",     str(meta.get("total_frames",     "—"))],
        ["Processed frames", str(meta.get("processed_frames", "—"))],
        ["Suspicious frames",str(meta.get("suspicious_frames","—"))],
        ["Processing time",  meta.get("elapsed",      "—")],
        ["Students detected",str(meta.get("student_count",    "—"))],
    ]
    t = Table(rows, colWidths=[5*cm, 12*cm])
    t.setStyle(_meta_table_style())
    story.append(t)


def _section_2_alert_summary(story, styles,
                               alert_summary: Dict[str, int]) -> None:
    _section_header(story, styles, "2. Alert summary")

    if not alert_summary:
        story.append(Paragraph("No alerts triggered.", styles["body"]))
        return

    total = sum(alert_summary.values())
    max_cnt = max(alert_summary.values()) or 1

    rows = [["Alert type", "Count", "Distribution"]]
    for event, cnt in sorted(alert_summary.items(), key=lambda x: -x[1]):
        bar_len = max(1, int((cnt / max_cnt) * 20))
        bar     = "█" * bar_len
        pct     = f"{cnt/total*100:.0f}%"
        rows.append([event, f"{cnt}  ({pct})", bar])

    rows.append(["TOTAL", str(total), ""])

    t = Table(rows, colWidths=[7*cm, 3*cm, 7*cm])
    t.setStyle(_summary_table_style(len(rows)))
    story.append(t)


def _section_3_timeline(story, styles,
                          timeline: List[Dict[str, Any]]) -> None:
    _section_header(story, styles, "3. Event timeline")

    if not timeline:
        story.append(Paragraph("No events recorded.", styles["body"]))
        return

    rows = [["Time", "Event", "Student"]]
    for ev in timeline:
        rows.append([
            ev.get("video_time", ""),
            ev.get("event",      ""),
            ev.get("student_id", ""),
        ])

    t = Table(rows, colWidths=[3*cm, 8.5*cm, 5.5*cm])
    t.setStyle(_timeline_table_style(len(rows)))
    story.append(t)


def _section_4_per_student(
    story, styles,
    student_scores: List[Tuple[str, float, str]],
    evidence_dir: str | Path,
    classifications: Optional[Dict[str, Any]],
) -> None:
    _section_header(story, styles, "4. Per-student profile")

    evidence_dir = Path(evidence_dir)
    risk_colors  = _RISK_COLORS

    for sid, score, risk in student_scores:
        story.append(Spacer(1, 0.3*cm))
        rc = risk_colors.get(risk, _GRAY)

        # Student header row
        hdr = Table(
            [[f"{sid}", f"Score: {score:.1f}", f"Risk: {risk}"]],
            colWidths=[4*cm, 5*cm, 8*cm],
        )
        hdr.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), rc),
            ("TEXTCOLOR",   (0, 0), (-1, 0), _WHITE),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0), 11),
            ("ALIGN",       (0, 0), (-1, 0), "LEFT"),
            ("LEFTPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING",  (0, 0), (-1, 0), 6),
            ("BOTTOMPADDING",(0, 0),(-1, 0), 6),
            ("ROWBACKGROUNDS",(0,0),(-1,-1),[rc]),
        ]))
        story.append(hdr)

        # Classifier verdict
        if classifications and sid in classifications:
            cl = classifications[sid]
            lbl = cl.label if hasattr(cl, "label") else str(cl)
            ml  = getattr(cl, "ml_label", None)
            verdict = f"Rule-based: {lbl}"
            if ml:
                verdict += f"   |   ML: {ml} ({getattr(cl,'ml_confidence',0):.0%})"
            story.append(Paragraph(verdict, styles["small_bold"]))

        # Event count table
        event_counts = {}
        if classifications and sid in classifications:
            event_counts = getattr(classifications[sid], "event_counts", {})
        if event_counts:
            ec_rows = [["Event", "Count"]] + [
                [k, str(v)] for k, v in sorted(event_counts.items(), key=lambda x: -x[1])
            ]
            et = Table(ec_rows, colWidths=[10*cm, 3*cm])
            et.setStyle(_mini_table_style(len(ec_rows)))
            story.append(et)
        else:
            story.append(Paragraph("No events in window.", styles["small"]))

        # Evidence images
        if _INCLUDE_EV and evidence_dir.exists():
            imgs = _collect_evidence_images(evidence_dir, sid)[:_MAX_EVIDENCE]
            if imgs:
                story.append(Spacer(1, 0.2*cm))
                story.append(Paragraph("Evidence frames:", styles["small_bold"]))
                img_row = []
                for img_path in imgs:
                    try:
                        img_row.append(Image(str(img_path), width=5.5*cm, height=3.5*cm))
                    except Exception:
                        pass
                if img_row:
                    it = Table([img_row], colWidths=[5.8*cm] * len(img_row))
                    story.append(it)

        story.append(HRFlowable(width="100%", thickness=0.3, color=_LGRAY))


def _section_5_classifier(story, styles,
                            classifications: Dict[str, Any]) -> None:
    _section_header(story, styles, "5. Cheating classifier output")

    rows = [["Student", "Rule label", "Rule score", "ML label", "ML confidence"]]
    for sid, cl in sorted(classifications.items()):
        label  = cl.label          if hasattr(cl, "label")          else "—"
        score  = f"{cl.rule_score:.1f}" if hasattr(cl, "rule_score") else "—"
        ml_lbl = cl.ml_label       if hasattr(cl, "ml_label") and cl.ml_label else "—"
        ml_con = f"{cl.ml_confidence:.0%}" if hasattr(cl, "ml_confidence") and cl.ml_label else "—"
        rows.append([sid, label, score, ml_lbl, ml_con])

    t = Table(rows, colWidths=[3*cm, 4*cm, 3*cm, 4*cm, 3*cm])
    ts = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), _BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0), _WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[_LGRAY, _WHITE]),
        ("FONTSIZE",     (0, 1), (-1,-1), 9),
        ("ALIGN",        (0, 0), (-1,-1), "CENTER"),
        ("GRID",         (0, 0), (-1,-1), 0.3, _GRAY),
        ("TOPPADDING",   (0, 0), (-1,-1), 4),
        ("BOTTOMPADDING",(0, 0), (-1,-1), 4),
    ])
    # Colour classifier labels
    for r, (sid, cl) in enumerate(sorted(classifications.items()), start=1):
        lbl = cl.label if hasattr(cl, "label") else ""
        c   = _CLASS_COLORS.get(lbl, _GRAY)
        ts.add("TEXTCOLOR", (1, r), (1, r), c)
        ts.add("FONTNAME",  (1, r), (1, r), "Helvetica-Bold")
    t.setStyle(ts)
    story.append(t)


def _section_6_analytics(story, styles,
                           room_metrics: Dict[str, float],
                           alert_summary: Dict[str, int]) -> None:
    _section_header(story, styles, "6. Room-level analytics")

    rows = [["Metric", "Value", "Status"]]
    thresholds = {
        "lookaway_rate":       ("Look-away rate",      0.45),
        "hand_activity_rate":  ("Hand activity rate",  0.60),
        "gesture_rate":        ("Gesture rate",        0.20),
        "phone_presence_rate": ("Phone presence rate", 0.08),
        "multiple_people_rate":("Multiple persons",    0.05),
    }
    for key, (label, thresh) in thresholds.items():
        val = room_metrics.get(key)
        if val is None:
            continue
        status = "ELEVATED" if val > thresh else "normal"
        rows.append([label, f"{val:.1%}", status])

    t = Table(rows, colWidths=[7*cm, 4*cm, 6*cm])
    ts = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), _TEAL),
        ("TEXTCOLOR",    (0, 0), (-1, 0), _WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[_LGRAY, _WHITE]),
        ("GRID",         (0, 0), (-1,-1), 0.3, _GRAY),
        ("TOPPADDING",   (0, 0), (-1,-1), 4),
        ("BOTTOMPADDING",(0, 0), (-1,-1), 4),
    ])
    for r in range(1, len(rows)):
        if rows[r][2] == "ELEVATED":
            ts.add("TEXTCOLOR", (2, r), (2, r), _RED)
            ts.add("FONTNAME",  (2, r), (2, r), "Helvetica-Bold")
    t.setStyle(ts)
    story.append(t)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _collect_evidence_images(evidence_dir: Path, sid: str) -> List[Path]:
    imgs = []
    for subfolder in evidence_dir.iterdir():
        if not subfolder.is_dir():
            continue
        for img in sorted(subfolder.glob(f"{sid}_*.jpg")):
            imgs.append(img)
    return sorted(imgs)[:_MAX_EVIDENCE]


def _build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title":      ParagraphStyle("title",      fontSize=18, fontName="Helvetica-Bold",
                                     spaceAfter=6, alignment=TA_CENTER),
        "section":    ParagraphStyle("section",    fontSize=13, fontName="Helvetica-Bold",
                                     spaceAfter=4, spaceBefore=10, textColor=_BLUE),
        "body":       ParagraphStyle("body",        fontSize=9,  fontName="Helvetica",
                                     spaceAfter=4),
        "small":      ParagraphStyle("small",       fontSize=8,  fontName="Helvetica",
                                     textColor=_GRAY),
        "small_bold": ParagraphStyle("small_bold",  fontSize=8,  fontName="Helvetica-Bold"),
    }


def _meta_table_style() -> TableStyle:
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), _TEAL),
        ("TEXTCOLOR",    (0, 0), (-1, 0), _WHITE),
        ("FONTNAME",     (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[_LGRAY, _WHITE]),
        ("GRID",         (0, 0), (-1,-1), 0.3, _GRAY),
        ("TOPPADDING",   (0, 0), (-1,-1), 4),
        ("BOTTOMPADDING",(0, 0), (-1,-1), 4),
    ])


def _summary_table_style(nrows: int) -> TableStyle:
    ts = TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), _BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0), _WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME",     (0, nrows-1), (-1, nrows-1), "Helvetica-Bold"),
        ("BACKGROUND",   (0, nrows-1), (-1, nrows-1), _LGRAY),
        ("FONTSIZE",     (0, 0), (-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,nrows-2),[_LGRAY, _WHITE]),
        ("GRID",         (0, 0), (-1,-1), 0.3, _GRAY),
        ("TOPPADDING",   (0, 0), (-1,-1), 4),
        ("BOTTOMPADDING",(0, 0), (-1,-1), 4),
        ("ALIGN",        (1, 0), (1,-1), "CENTER"),
    ])
    return ts


def _timeline_table_style(nrows: int) -> TableStyle:
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), _GRAY),
        ("TEXTCOLOR",    (0, 0), (-1, 0), _WHITE),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[_LGRAY, _WHITE]),
        ("GRID",         (0, 0), (-1,-1), 0.3, _GRAY),
        ("TOPPADDING",   (0, 0), (-1,-1), 3),
        ("BOTTOMPADDING",(0, 0), (-1,-1), 3),
        ("ALIGN",        (0, 0), (0,-1), "CENTER"),
    ])


def _mini_table_style(nrows: int) -> TableStyle:
    return TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), _LGRAY),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1,-1), 8),
        ("GRID",         (0, 0), (-1,-1), 0.3, _GRAY),
        ("TOPPADDING",   (0, 0), (-1,-1), 2),
        ("BOTTOMPADDING",(0, 0), (-1,-1), 2),
    ])


def _header_footer(canvas, doc) -> None:
    canvas.saveState()
    # Header line
    canvas.setStrokeColor(_GRAY)
    canvas.setLineWidth(0.5)
    canvas.line(2*cm, A4[1] - 1.5*cm, A4[0] - 2*cm, A4[1] - 1.5*cm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(_GRAY)
    canvas.drawString(2*cm, A4[1] - 1.3*cm, "AI Proctoring System — ALTIMA Security Technologies")
    canvas.drawRightString(A4[0] - 2*cm, A4[1] - 1.3*cm,
                           datetime.now().strftime("%Y-%m-%d"))
    # Footer
    canvas.line(2*cm, 1.2*cm, A4[0] - 2*cm, 1.2*cm)
    canvas.drawString(2*cm, 0.8*cm, "CONFIDENTIAL — for authorised examination staff only")
    canvas.drawRightString(A4[0] - 2*cm, 0.8*cm, f"Page {doc.page}")
    canvas.restoreState()


# ── Plain-text fallback (if ReportLab missing) ────────────────────────────────

def _plain_text_fallback(path: Path, meta, summary, timeline, scores) -> None:
    lines = ["=" * 62, "  AI PROCTORING REPORT", "=" * 62, ""]
    for k, v in meta.items():
        lines.append(f"  {k}: {v}")
    lines += ["", "ALERT SUMMARY", "-" * 30]
    for ev, cnt in sorted(summary.items(), key=lambda x: -x[1]):
        lines.append(f"  {ev:<35} {cnt}")
    lines += ["", "TIMELINE", "-" * 30]
    for ev in timeline:
        lines.append(f"  [{ev.get('video_time','')}]  {ev.get('event',''):<32}  {ev.get('student_id','')}")
    lines += ["", "RISK SUMMARY", "-" * 30]
    for sid, score, risk in scores:
        lines.append(f"  {sid:<8} score={score:>6.1f}  [{risk}]")
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Plain-text fallback report saved → %s", path)
