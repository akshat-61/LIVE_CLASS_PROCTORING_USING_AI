import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

def _has_display() -> bool:
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

FRAMES_DIR    = "frames"
ANNOTATED_DIR = "annotated"
OUTPUT_DIR    = "evidence_output"

os.makedirs(FRAMES_DIR,    exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,    exist_ok=True)

def parse_args():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    video_path = sys.argv[1]
    exam_id    = sys.argv[2] if len(sys.argv) > 2 else None
    room_id    = sys.argv[3] if len(sys.argv) > 3 else None
    return video_path, exam_id, room_id

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
_video_ts   = [0.0]   


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

def extract_frames(video_path: str, show_preview: bool = True) -> tuple[list[str], float, float]:
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

        if idx % 300 == 0:
            pct     = idx / total_frames * 100 if total_frames else 0
            elapsed = time.time() - t0
            fps_est = idx / elapsed if elapsed > 0 else 0
            print(f"  Extracted {idx}/{total_frames}  ({pct:.0f}%)  "
                  f"|  {fps_est:.0f} frames/s")

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

def _compute_adaptive_zone_radius(positions: dict) -> int:
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
    import mediapipe as mp

    snap_seconds   = [1.0, 2.0, 3.0]
    iw, ih         = ae.AI_FRAME_SIZE
    accumulated    = {}

    snap_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=20,
        refine_landmarks=True,
        min_detection_confidence=0.15,
        min_tracking_confidence=0.15,
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

        for img_rgb, frame_pos in snap_frames_rgb:
            if frame_pos:
                ae.learner.on_calibration_frame(
                    img_rgb, frame_pos,
                    snap_face_mesh, ae.pose_detector, iw, ih,
                )

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

    radius           = _compute_adaptive_zone_radius(accumulated)
    ae.SEAT_ZONE_RADIUS = radius
    print(f"  📐 Adaptive zone radius : {radius}px")

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

def _frame_is_suspicious(annotated: np.ndarray) -> bool:
    if annotated is None or annotated.shape[0] < 10:
        return False
    b = int(annotated[25, 5, 0])
    g = int(annotated[25, 5, 1])
    r = int(annotated[25, 5, 2])
    return r > g and r > 60


def analyse_frames(
    frame_paths : list,
    src_fps     : float,
    duration_s  : float,
    ae,
    show_preview: bool = True,
) -> tuple[int, int]:
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

        _video_ts[0] = (i / src_fps) if src_fps > 0 else 0.0

        annotated = run_ai_on_frame(img)

        if _frame_is_suspicious(annotated):
            out_path = os.path.join(ANNOTATED_DIR, os.path.basename(path))
            cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            suspicious_count += 1

        if show_preview:
            cv2.imshow("AI Analysis", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("\n[!] Analysis quit early.")
                break

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

def process_video(video_path, exam_id=None, room_id=None, show_preview=True):
    if not os.path.exists(video_path):
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)

    t_total = time.time()

    import ai_engine as ae
    import core.ai_engine as _ae_core  # patch the real module, not the shim

    if exam_id:
        _ae_core.EXAM_ID = exam_id
    if room_id:
        _ae_core.ROOM_ID = room_id

    _ae_core.send_event_async = make_patched_sender(_ae_core)
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