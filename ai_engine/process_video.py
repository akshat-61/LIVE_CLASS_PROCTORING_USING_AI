"""
=============================================================
  AI Proctoring System — Batch Frame Processor
  v3 — Extract-Then-Analyse Mode
=============================================================

  HOW IT WORKS:
  ─────────────
  PHASE 1 — EXTRACTION
    Read every frame from the input video and save it to
    frames/frame_000001.jpg, frames/frame_000002.jpg, …
    No AI is run during this phase — pure fast extraction.

  PHASE 2 — CALIBRATION
    Run multi-snapshot calibration on frames at t=1s, 2s, 3s
    to lock student IDs and seat zones before analysis begins.

  PHASE 3 — ANALYSIS
    Run run_ai_on_frame() on every saved frame.
    Only frames where AI detects something suspicious are saved
    to annotated/frame_000001.jpg, …
    All events are logged to logs/alerts_<date>.json.

  Run:
      python process_video.py <video_path> [exam_id] [room_id]

  Examples:
      python process_video.py exam_footage.mp4
      python process_video.py footage.mp4 EXAM_007 ROOM_B

  Output:
      frames/                    — all extracted raw frames
      annotated/                 — suspicious frames with AI overlay
      logs/alerts_<date>.json    — alert log
      evidence/                  — saved evidence crops
      evidence_output/<n>_report.txt — full timeline report

  Controls during extraction preview:
      Q / ESC  →  quit early
=============================================================
"""

import os
import sys

# Fix OpenCV window rendering on Wayland (Ubuntu/GNOME).
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

# Auto-detect whether a display is available.
# If DISPLAY and WAYLAND_DISPLAY are both unset we are running headless
# (SSH session, cron job, server) — disable all cv2.imshow calls so
# OpenCV never tries to open a window and crashes with the GTK error.
def _has_display() -> bool:
    """Return True only if a real GUI display is reachable."""
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        try:
            import cv2 as _cv2
            # Quick probe — will raise if GTK/Cocoa not available
            _cv2.namedWindow("__probe__", _cv2.WINDOW_NORMAL)
            _cv2.destroyWindow("__probe__")
            return True
        except Exception:
            return False
    return False

_DISPLAY_AVAILABLE = _has_display()
if not _DISPLAY_AVAILABLE:
    print("[INFO] No display detected — running in headless mode (no preview windows)")

import cv2
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ─── Directories ──────────────────────────────────────────────────────────────

FRAMES_DIR    = "frames"
ANNOTATED_DIR = "annotated"
OUTPUT_DIR    = "evidence_output"

os.makedirs(FRAMES_DIR,    exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,    exist_ok=True)

# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    video_path = sys.argv[1]
    exam_id    = sys.argv[2] if len(sys.argv) > 2 else None
    room_id    = sys.argv[3] if len(sys.argv) > 3 else None
    return video_path, exam_id, room_id

# ─── Alert interceptor (for report generation) ────────────────────────────────

class AlertInterceptor:
    def __init__(self):
        self.events = []

    def record(self, event_type, student_id, video_ts):
        self.events.append({
            "video_time": str(timedelta(seconds=int(video_ts))),
            "event":      event_type,
            "student_id": student_id,
            "wall_clock": datetime.now().strftime("%H:%M:%S"),
        })

    def summary(self):
        from collections import Counter
        return dict(sorted(
            Counter(e["event"] for e in self.events).items(),
            key=lambda x: -x[1]
        ))


interceptor = AlertInterceptor()
_video_ts   = [0.0]   # mutable container so nested closure can write to it


def make_patched_sender(ae):
    import threading, requests

    def _patched(event_type, student_id=None, distance=None):
        interceptor.record(event_type, student_id, _video_ts[0])
        ts_str = str(timedelta(seconds=int(_video_ts[0])))
        print(f"  🚨 [{ts_str}]  {event_type:<32}  student: {student_id}")

        def task():
            payload = {
                "examId": ae.EXAM_ID,
                "roomId": ae.ROOM_ID,
                "events": [{
                    "type":       event_type,
                    "timestamp":  time.time(),
                    "confidence": 0.9,
                    "trackId":    student_id,
                    "distance":   distance,
                    "video_time": _video_ts[0],
                }],
            }
            try:
                requests.post(ae.ALERT_URL, json=payload, timeout=1)
            except Exception:
                pass
        threading.Thread(target=task, daemon=True).start()

    return _patched

# ─── Phase 1: Frame Extraction ────────────────────────────────────────────────

def extract_frames(video_path: str, show_preview: bool = True) -> tuple[list[str], float, float]:
    """
    Read every frame from the video and save to frames/frame_XXXXXX.jpg.

    Returns:
        frame_paths  — ordered list of saved file paths
        src_fps      — original video FPS
        duration_s   — total video duration in seconds
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    src_fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / src_fps
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("\n" + "=" * 62)
    print("  PHASE 1 — FRAME EXTRACTION")
    print("=" * 62)
    print(f"  File       : {video_path}")
    print(f"  Resolution : {w}×{h}  @  {src_fps:.1f}fps")
    print(f"  Duration   : {str(timedelta(seconds=int(duration_s)))}")
    print(f"  Frames     : {total_frames}")
    print(f"  Saving to  : {FRAMES_DIR}/")
    print("=" * 62)

    show_preview = show_preview and _DISPLAY_AVAILABLE
    if show_preview:
        cv2.namedWindow("Extraction Preview", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Extraction Preview", 960, 540)

    frame_paths = []
    idx         = 0
    t0          = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        idx += 1
        path = os.path.join(FRAMES_DIR, f"frame_{idx:06d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        frame_paths.append(path)

        # Progress every 300 frames
        if idx % 300 == 0:
            pct     = idx / total_frames * 100 if total_frames else 0
            elapsed = time.time() - t0
            fps_est = idx / elapsed if elapsed > 0 else 0
            print(f"  Extracted {idx}/{total_frames}  ({pct:.0f}%)  "
                  f"|  {fps_est:.0f} frames/s")

        # Live preview (every 10th frame to avoid slowing extraction)
        if show_preview and idx % 10 == 0:
            cv2.imshow("Extraction Preview", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("\n[!] Extraction quit early.")
                break

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\n  ✅ Extracted {len(frame_paths)} frames in {elapsed:.1f}s  "
          f"({len(frame_paths)/elapsed:.0f} frames/s)")
    print(f"  Saved to : {FRAMES_DIR}/\n")

    return frame_paths, src_fps, duration_s

# ─── Phase 2: Snapshot Calibration ────────────────────────────────────────────

def _compute_adaptive_zone_radius(positions: dict) -> int:
    """Camera-agnostic seat zone radius from nearest-neighbour distances."""
    if len(positions) < 2:
        return 35
    pts      = list(positions.values())
    nn_dists = []
    for i, p in enumerate(pts):
        others = [np.linalg.norm(np.array(p) - np.array(q))
                  for j, q in enumerate(pts) if j != i]
        nn_dists.append(min(others))
    median_nn = float(np.median(nn_dists))
    return max(18, min(60, int(median_nn * 0.35)))


def snapshot_calibrate(frame_paths: list, est_fps: float, ae) -> None:
    """
    Multi-snapshot calibration directly from saved frame files.

    Scans frames at t≈1s, 2s, 3s (±0.5s window each) to:
      1. Detect all persons via YOLO person_model
      2. Feed FaceMesh data into AdaptiveLearner so baselines are warm
      3. Compute adaptive seat-zone radius
      4. Assign student IDs and lock seat zones

    This mirrors snapshot_calibrate() from the old process_video.py but
    operates on frame_paths instead of re-opening the video file.
    """
    import mediapipe as mp

    snap_seconds   = [1.0, 2.0, 3.0]
    iw, ih         = ae.AI_FRAME_SIZE
    accumulated    = {}

    snap_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=20,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    print("=" * 62)
    print("  PHASE 2 — SNAPSHOT CALIBRATION")
    print("=" * 62)

    for snap_t in snap_seconds:
        target_idx = int(est_fps * snap_t)
        win_start  = max(0, int(est_fps * (snap_t - 0.5)))
        win_end    = min(len(frame_paths) - 1, int(est_fps * (snap_t + 0.5)))

        if target_idx >= len(frame_paths):
            print(f"  [SKIP] t={snap_t}s — beyond frame count")
            continue

        snap_positions  = {}
        snap_frames_rgb = []

        step = max(1, int(est_fps / 6))
        for fi in range(win_start, win_end + 1, step):
            img = cv2.imread(frame_paths[fi])
            if img is None:
                continue
            snap    = cv2.resize(img, ae.AI_FRAME_SIZE)
            img_rgb = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)

            track_res = ae.person_model.track(
                snap, persist=True, verbose=False,
                imgsz=640, max_det=40, conf=ae.CUSTOM_CONF_THRESHOLD
            )
            frame_positions = {}
            if track_res and track_res[0].boxes is not None:
                for box in track_res[0].boxes:
                    if float(box.conf[0]) < ae.CUSTOM_CONF_THRESHOLD:
                        continue
                    if int(box.cls[0]) != ae.CUSTOM_CLASS["PERSON"]:
                        continue
                    if box.id is None:
                        continue
                    tid = int(box.id[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    frame_positions[tid] = (cx, cy)
                    if tid not in snap_positions:
                        snap_positions[tid] = (cx, cy)
                    else:
                        ox, oy = snap_positions[tid]
                        snap_positions[tid] = (int(ox*0.6+cx*0.4), int(oy*0.6+cy*0.4))

            snap_frames_rgb.append((img_rgb, dict(frame_positions)))

        # Feed face-mesh calibration
        for img_rgb, frame_pos in snap_frames_rgb:
            if frame_pos:
                ae.learner.on_calibration_frame(
                    img_rgb, frame_pos,
                    snap_face_mesh, ae.pose_detector, iw, ih,
                )

        # Merge into accumulated positions
        new_count = 0
        for tid, pos in snap_positions.items():
            if tid not in accumulated:
                accumulated[tid] = pos
                new_count += 1
            else:
                ox, oy = accumulated[tid]
                accumulated[tid] = (int(ox*0.7+pos[0]*0.3), int(oy*0.7+pos[1]*0.3))

        print(f"  t={snap_t}s : {len(snap_positions)} persons  (+{new_count} new)")

    snap_face_mesh.close()

    if not accumulated:
        print("  [WARN] No persons detected — running without seat zones")
        ae.locked = True
        return

    # Adaptive zone radius
    radius           = _compute_adaptive_zone_radius(accumulated)
    ae.SEAT_ZONE_RADIUS = radius
    print(f"  📐 Adaptive zone radius : {radius}px")

    # Assign IDs and lock
    ae.student_id_map.update(ae.assign_student_ids(accumulated))
    ae.learner.on_lock(ae.student_id_map)
    ae.init_seat_zones(accumulated)
    ae.learner.set_seat_positions(ae.seat_positions)

    ae.locked         = True
    ae.stable_counter = ae.STABILITY_FRAMES
    ae._lock_frame    = 0
    ae._calib_accumulated_positions.update(accumulated)

    n_cal = sum(
        1 for sid in ae.student_id_map.values()
        if ae.learner.is_calibrated(sid)
    )
    print(f"  🔒 Locked {len(ae.student_id_map)} students  |  "
          f"pre-calibrated: {n_cal}/{len(ae.student_id_map)}")

    # Save seat-map snapshot from first frame
    if frame_paths:
        vis = cv2.imread(frame_paths[0])
        if vis is not None:
            vis = cv2.resize(vis, ae.AI_FRAME_SIZE)
            for tid, (cx, cy) in accumulated.items():
                sid = ae.student_id_map.get(tid, "?")
                cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(vis, sid, (cx - 20, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                cv2.circle(vis, (cx, cy), ae.SEAT_ZONE_RADIUS, (255, 200, 0), 1)
            snap_path = os.path.join(OUTPUT_DIR, "snapshot_calibration.jpg")
            cv2.imwrite(snap_path, vis)
            print(f"  Seat-map snapshot → {snap_path}")

    print("=" * 62 + "\n")

# ─── Phase 3: AI Analysis ──────────────────────────────────────────────────────

def _frame_is_suspicious(annotated: np.ndarray) -> bool:
    """
    Detect whether ai_engine flagged an alert in this frame.

    run_ai_on_frame draws a green banner (0,130,0) for All Clear
    and a red banner (0,0,200) for any alert, blended with the frame.
    We sample the banner region (row 25, col 5) and check which
    channel dominates after blending.
    """
    if annotated is None or annotated.shape[0] < 10:
        return False
    b = int(annotated[25, 5, 0])
    g = int(annotated[25, 5, 1])
    r = int(annotated[25, 5, 2])
    return r > g and r > 60   # red dominant → alert banner


def analyse_frames(
    frame_paths : list,
    src_fps     : float,
    duration_s  : float,
    ae,
    show_preview: bool = True,
) -> tuple[int, int]:
    """
    Run run_ai_on_frame() on every saved frame.
    Saves annotated output only for suspicious frames.

    Returns:
        processed_count   — total frames processed
        suspicious_count  — frames saved to annotated/
    """
    from ai_engine import run_ai_on_frame

    total = len(frame_paths)

    print("=" * 62)
    print("  PHASE 3 — AI ANALYSIS")
    print("=" * 62)
    print(f"  Frames to process : {total}")
    print(f"  Suspicious saved  : {ANNOTATED_DIR}/")
    print("=" * 62)

    show_preview = show_preview and _DISPLAY_AVAILABLE
    if show_preview:
        cv2.namedWindow("AI Analysis", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI Analysis", 960, 540)

    suspicious_count = 0
    t0               = time.time()

    for i, path in enumerate(frame_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        # Update video timestamp for alert interceptor
        _video_ts[0] = (i / src_fps) if src_fps > 0 else 0.0

        annotated = run_ai_on_frame(img)

        if _frame_is_suspicious(annotated):
            out_path = os.path.join(ANNOTATED_DIR, os.path.basename(path))
            cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            suspicious_count += 1

        # Preview every processed frame
        if show_preview:
            cv2.imshow("AI Analysis", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("\n[!] Analysis quit early.")
                break

        # Progress log every 500 frames
        if (i + 1) % 500 == 0:
            pct     = (i + 1) / total * 100
            elapsed = time.time() - t0
            fps_est = (i + 1) / elapsed if elapsed > 0 else 0
            ts_str  = str(timedelta(seconds=int(_video_ts[0])))
            print(f"  [{ts_str}]  {i+1}/{total}  ({pct:.0f}%)  "
                  f"|  suspicious: {suspicious_count}  "
                  f"|  {fps_est:.1f} fps")

    if show_preview:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\n  ✅ Processed {total} frames in {elapsed:.1f}s  "
          f"({total/elapsed:.1f} fps)")
    print(f"  Suspicious frames : {suspicious_count}  →  {ANNOTATED_DIR}/\n")

    return total, suspicious_count

# ─── Report ────────────────────────────────────────────────────────────────────

def write_report(
    video_path       : str,
    total_frames     : int,
    processed_frames : int,
    suspicious_frames: int,
    src_fps          : float,
    duration_s       : float,
    elapsed_real     : float,
):
    stem  = Path(video_path).stem
    rpath = os.path.join(OUTPUT_DIR, f"{stem}_report.txt")
    summary = interceptor.summary()
    events  = interceptor.events

    lines = [
        "=" * 62,
        "  AI PROCTORING — BATCH ANALYSIS REPORT",
        "=" * 62,
        f"  Video          : {video_path}",
        f"  Processed at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  FPS            : {src_fps:.1f}   Duration: {str(timedelta(seconds=int(duration_s)))}",
        f"  Frames         : {processed_frames} processed / {total_frames} total",
        f"  Suspicious     : {suspicious_frames} frames saved to {ANNOTATED_DIR}/",
        f"  Real time      : {elapsed_real:.1f}s",
        "",
        "─" * 62,
        "  ALERT SUMMARY",
        "─" * 62,
    ]

    if not summary:
        lines.append("  No alerts triggered.")
    else:
        for ev, cnt in summary.items():
            lines.append(f"  {ev:<35}  {'█'*min(cnt, 35)} ({cnt})")

    # FIX 9 (v12): student risk summary section
    # Pull scores from ai_engine's score_engine after analysis is complete
    try:
        import ai_engine as _ae
        all_scores = _ae.score_engine.get_all_scores()
        risky = {sid: s for sid, s in all_scores.items() if s >= 5}
    except Exception:
        risky = {}

    if risky:
        lines += ["", "─" * 62, "  STUDENT RISK SUMMARY", "─" * 62]
        lines.append(f"  {'Student':<10}  {'Score':>7}  {'Risk Level':<12}  Events")
        lines.append(f"  {'─'*8}  {'─'*7}  {'─'*12}  {'─'*20}")
        for sid, score in risky.items():
            if score >= 80:
                level = "CRITICAL"
            elif score >= 40:
                level = "HIGH"
            elif score >= 15:
                level = "MEDIUM"
            else:
                level = "LOW"
            student_events = [e["event"] for e in events if e.get("student_id") == sid
                              or (e.get("student_id") or "").startswith(sid)]
            event_str = ", ".join(dict.fromkeys(student_events))[:35]  # deduplicated
            lines.append(f"  {sid:<10}  {score:>7.1f}  {level:<12}  {event_str}")

    lines += ["", "─" * 62, "  TIMELINE", "─" * 62]
    for ev in events:
        lines.append(
            f"  [{ev['video_time']:>8}]  {ev['event']:<32}  "
            f"student: {ev['student_id']}"
        )
    lines += ["", "=" * 62]

    with open(rpath, "w") as f:
        f.write("\n".join(lines))
    print(f"  📄 Report → {rpath}")
    return rpath

# ─── Main ──────────────────────────────────────────────────────────────────────

def process_video(
    video_path   : str,
    exam_id      : str  = None,
    room_id      : str  = None,
    show_preview : bool = True,
):
    if not os.path.exists(video_path):
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)

    t_total = time.time()

    # ── Import and patch ai_engine ────────────────────────────────────────
    import ai_engine as ae
    if exam_id: ae.EXAM_ID = exam_id
    if room_id: ae.ROOM_ID = room_id
    ae.send_event_async = make_patched_sender(ae)
    ae.reset_session()

    # ── Phase 1: Extract all frames ───────────────────────────────────────
    frame_paths, src_fps, duration_s = extract_frames(video_path, show_preview)
    total_frames = len(frame_paths)

    if total_frames == 0:
        print("[ERROR] No frames extracted.")
        sys.exit(1)

    # ── Phase 2: Snapshot calibration ─────────────────────────────────────
    snapshot_calibrate(frame_paths, src_fps, ae)

    # ── Phase 3: AI analysis on all frames ────────────────────────────────
    processed, suspicious = analyse_frames(
        frame_paths, src_fps, duration_s, ae, show_preview
    )

    # ── Report ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    report_path = write_report(
        video_path, total_frames, processed, suspicious,
        src_fps, duration_s, elapsed
    )

    ae.shutdown()

    # ── Summary ────────────────────────────────────────────────────────────
    print("=" * 62)
    print("  ALL DONE")
    print("=" * 62)
    print(f"  Raw frames     : {total_frames}  →  {FRAMES_DIR}/")
    print(f"  Suspicious     : {suspicious}  →  {ANNOTATED_DIR}/")
    print(f"  Alerts total   : {len(interceptor.events)}")
    print(f"  Total time     : {elapsed:.1f}s")
    print(f"  Seat map       : {OUTPUT_DIR}/snapshot_calibration.jpg")
    print(f"  Report         : {report_path}")
    print("=" * 62)

    summary = interceptor.summary()
    if summary:
        print("\n  Alert breakdown:")
        for ev, cnt in summary.items():
            print(f"    {ev:<35}  {'█'*min(cnt, 40)} ({cnt})")
    else:
        print("\n  ✅  No alerts triggered.")
    print()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    video_path, exam_id, room_id = parse_args()
    process_video(
        video_path   = video_path,
        exam_id      = exam_id,
        room_id      = room_id,
        show_preview = True,
    )