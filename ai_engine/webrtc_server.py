import asyncio
import threading
import time
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
import ai_engine
from ai_engine import run_ai_on_frame

# ─── Configuration ────────────────────────────────────────────────────────────

EXAM_DURATION_SECONDS = 30 * 60        # 30 minutes — change as needed
FRAMES_DIR            = "frames"       # raw frames saved here: frames/frame_0001.jpg
ANNOTATED_DIR         = "annotated"    # suspicious annotated frames saved here
os.makedirs(FRAMES_DIR,    exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# ─── State ────────────────────────────────────────────────────────────────────

pcs = set()

# Frame collection state
_collected_frames  = []        # list of saved file paths (in order)
_frame_index       = 0         # global counter for naming frame files
_collection_lock   = threading.Lock()
_collection_active = False     # True once first WebRTC track arrives
_exam_start_time   = None      # wall-clock time when collection started
_exam_ended        = False     # True once timer fires or session ends

# ─── Display thread (raw only — no AI overlay during collection) ──────────────

_live_frame    = None
_display_lock  = threading.Lock()
_display_stop  = threading.Event()


def _display_thread():
    """Shows the raw incoming video. No AI processing during collection."""
    while not _display_stop.is_set():
        with _display_lock:
            frame = _live_frame
        if frame is not None:
            cv2.imshow("Proctoring — Recording (Raw)", frame)
        # ~30fps display loop
        if cv2.waitKey(33) & 0xFF == ord("q"):
            _display_stop.set()
            break
    cv2.destroyAllWindows()


threading.Thread(target=_display_thread, daemon=True).start()

# ─── Exam timer ───────────────────────────────────────────────────────────────

def _start_exam_timer():
    """Fires once after EXAM_DURATION_SECONDS, triggers post-exam AI analysis."""
    global _exam_ended
    print(f"[TIMER] Exam timer started — {EXAM_DURATION_SECONDS}s remaining")

    def _timer_callback():
        global _exam_ended
        if not _exam_ended:
            _exam_ended = True
            print("[TIMER] Exam duration reached — starting AI analysis")
            _run_post_exam_analysis()

    timer = threading.Timer(EXAM_DURATION_SECONDS, _timer_callback)
    timer.daemon = True
    timer.start()
    return timer


_exam_timer = None  # holds the active Timer object so it can be cancelled

# ─── CORS Middleware ───────────────────────────────────────────────────────────

@web.middleware
async def cors_middleware(request, handler):
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        response = await handler(request)
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# ─── WebRTC Offer Handler ──────────────────────────────────────────────────────

async def offer(request):
    global _collection_active, _exam_start_time, _exam_timer

    params    = await request.json()
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[WebRTC] Connection state: {pc.connectionState}")
        if pc.connectionState in ["failed", "closed"]:
            print("[WebRTC] Client disconnected")
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        global _collection_active, _exam_start_time, _exam_timer
        print(f"[WebRTC] Track received: {track.kind}")
        if track.kind == "video":
            # Start collection + timer on first video track
            if not _collection_active:
                _collection_active = True
                _exam_start_time   = time.time()
                _exam_timer        = _start_exam_timer()
                print("[COLLECT] Frame collection started")
            asyncio.get_event_loop().create_task(collect_frames(track))

    await pc.setRemoteDescription(offer_sdp)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response({
        "sdp":  pc.localDescription.sdp,
        "type": pc.localDescription.type,
    })

async def collect_frames(track):
    global _frame_index, _live_frame

    loop = asyncio.get_event_loop()

    while not _exam_ended:
        try:
            frame = await track.recv()
        except Exception as e:
            print("[WebRTC] Track ended:", e)
            break

        # Convert to numpy array (BGR)
        img = frame.to_ndarray(format="bgr24")

        # Always update the live display
        with _display_lock:
            _live_frame = img

        # Save frame to disk
        with _collection_lock:
            _frame_index += 1
            idx = _frame_index

        filename = os.path.join(FRAMES_DIR, f"frame_{idx:06d}.jpg")

        # Write in a thread so we don't block the async event loop
        await loop.run_in_executor(None, _save_frame, filename, img)

        with _collection_lock:
            _collected_frames.append(filename)

        if idx % 300 == 0:
            print(f"[COLLECT] {idx} frames saved so far")

    print(f"[COLLECT] Collection stopped — {_frame_index} total frames saved")


def _save_frame(path: str, img: np.ndarray):
    """Blocking JPEG write — runs in a thread pool executor."""
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])

# ─── Post-Exam AI Analysis ─────────────────────────────────────────────────────

_analysis_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai_analysis")


def _run_post_exam_analysis():
    _analysis_executor.submit(_analyse_all_frames)


def _analyse_all_frames():
    print("\n" + "=" * 60)
    print("  POST-EXAM AI ANALYSIS STARTING")
    print("=" * 60)

    with _collection_lock:
        frame_paths = list(_collected_frames)

    if not frame_paths:
        print("[ANALYSIS] No frames collected — nothing to analyse")
        return

    total = len(frame_paths)
    print(f"[ANALYSIS] {total} frames to process")
    ai_engine.reset_session()
    if _exam_start_time:
        elapsed  = time.time() - _exam_start_time
        est_fps  = total / elapsed if elapsed > 0 else 25.0
    else:
        est_fps = 25.0

    print(f"[ANALYSIS] Estimated collection FPS: {est_fps:.1f}")
    _snapshot_calibrate_from_frames(frame_paths, est_fps)

    # ── Step 2: Process every frame ────────────────────────────────────────
    suspicious_count = 0
    t0 = time.time()

    for i, path in enumerate(frame_paths):
        img = cv2.imread(path)
        if img is None:
            continue

        annotated = run_ai_on_frame(img)
        if _frame_is_suspicious(annotated):
            out_path = os.path.join(ANNOTATED_DIR, os.path.basename(path))
            cv2.imwrite(out_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            suspicious_count += 1
        if (i + 1) % 500 == 0:
            pct     = (i + 1) / total * 100
            elapsed = time.time() - t0
            fps_est = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"[ANALYSIS] {i+1}/{total}  ({pct:.0f}%)  "
                  f"|  suspicious: {suspicious_count}  "
                  f"|  {fps_est:.1f} fps")

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Frames processed  : {total}")
    print(f"  Suspicious saved  : {suspicious_count}  →  {ANNOTATED_DIR}/")
    print(f"  Alert log         : logs/")
    print(f"  Time taken        : {elapsed:.1f}s")
    print("=" * 60 + "\n")

    ai_engine.shutdown()


def _snapshot_calibrate_from_frames(frame_paths: list, est_fps: float):
    import mediapipe as mp

    snap_seconds   = [1.0, 2.0, 3.0]
    iw, ih         = ai_engine.AI_FRAME_SIZE
    accumulated    = {}

    snap_face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=20,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    print("[CALIB] Multi-snapshot calibration from saved frames…")

    for snap_t in snap_seconds:
        target_idx = int(est_fps * snap_t)
        win_start  = max(0, int(est_fps * (snap_t - 0.5)))
        win_end    = min(len(frame_paths) - 1, int(est_fps * (snap_t + 0.5)))

        if target_idx >= len(frame_paths):
            print(f"  [SKIP] t={snap_t}s — beyond frame count")
            continue

        snap_positions  = {}
        snap_frames_rgb = []

        # Step through the window
        step = max(1, int(est_fps / 6))
        for fi in range(win_start, win_end + 1, step):
            img = cv2.imread(frame_paths[fi])
            if img is None:
                continue
            snap = cv2.resize(img, ai_engine.AI_FRAME_SIZE)
            img_rgb = cv2.cvtColor(snap, cv2.COLOR_BGR2RGB)

            track_res = ai_engine.person_model.track(
                snap, persist=True, verbose=False,
                imgsz=640, max_det=40, conf=ai_engine.CUSTOM_CONF_THRESHOLD
            )
            frame_positions = {}
            if track_res and track_res[0].boxes is not None:
                for box in track_res[0].boxes:
                    if float(box.conf[0]) < ai_engine.CUSTOM_CONF_THRESHOLD:
                        continue
                    if int(box.cls[0]) != ai_engine.CUSTOM_CLASS["PERSON"]:
                        continue
                    if box.id is None:
                        continue
                    tid = int(box.id[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    frame_positions[tid] = (cx, cy)
                    if tid not in snap_positions:
                        snap_positions[tid] = (cx, cy)
                    else:
                        ox, oy = snap_positions[tid]
                        snap_positions[tid] = (int(ox*0.6+cx*0.4), int(oy*0.6+cy*0.4))

            snap_frames_rgb.append((img_rgb, dict(frame_positions)))

        # Feed face-mesh calibration frames
        for img_rgb, frame_pos in snap_frames_rgb:
            if frame_pos:
                ai_engine.learner.on_calibration_frame(
                    img_rgb, frame_pos,
                    snap_face_mesh, ai_engine.pose_detector, iw, ih,
                )

        # Merge into accumulated
        new_count = 0
        for tid, pos in snap_positions.items():
            if tid not in accumulated:
                accumulated[tid] = pos
                new_count += 1
            else:
                ox, oy = accumulated[tid]
                accumulated[tid] = (int(ox*0.7+pos[0]*0.3), int(oy*0.7+pos[1]*0.3))

        print(f"  t={snap_t}s: {len(snap_positions)} persons  (+{new_count} new)")

    snap_face_mesh.close()

    if not accumulated:
        print("[CALIB] No persons detected — running without seat zones")
        ai_engine.locked = True
        return

    # Assign IDs and lock seat zones
    ai_engine.student_id_map.update(ai_engine.assign_student_ids(accumulated))
    ai_engine.learner.on_lock(ai_engine.student_id_map)
    ai_engine.init_seat_zones(accumulated)
    ai_engine.learner.set_seat_positions(ai_engine.seat_positions)
    ai_engine.locked         = True
    ai_engine.stable_counter = ai_engine.STABILITY_FRAMES
    ai_engine._lock_frame    = 0

    n_cal = sum(
        1 for sid in ai_engine.student_id_map.values()
        if ai_engine.learner.is_calibrated(sid)
    )
    print(f"  ✅ Locked {len(ai_engine.student_id_map)} students  |  "
          f"pre-calibrated: {n_cal}/{len(ai_engine.student_id_map)}\n")


def _frame_is_suspicious(annotated: np.ndarray) -> bool:
    """
    Determine if ai_engine flagged something in this frame.

    run_ai_on_frame draws a green banner (0,130,0) for All Clear and a red
    banner (0,0,200) for alerts. We sample the top-left pixel of the banner
    row (y=0..50) and check if it is predominantly green vs red.

    Returns True  → suspicious (red/orange banner → save this frame)
    Returns False → all clear  (green banner → skip)
    """
    if annotated is None or annotated.shape[0] < 10:
        return False

    # Sample the banner region: row 25, column 5  (centre-left of banner)
    b, g, r = int(annotated[25, 5, 0]), int(annotated[25, 5, 1]), int(annotated[25, 5, 2])

    # All-clear banner is drawn with color (0, 130, 0) → green channel dominant
    # Alert banner is drawn with color (0, 0, 200) → red channel dominant
    # Allow some blending tolerance (cv2.addWeighted blends with the frame)
    if r > g and r > 60:
        return True   # red/alert banner
    return False      # green/clear banner

# ─── Cleanup ──────────────────────────────────────────────────────────────────

async def on_shutdown(app):
    global _exam_ended
    _display_stop.set()

    # If the server shuts down before the timer fires, trigger analysis now
    if _collection_active and not _exam_ended:
        _exam_ended = True
        print("[SHUTDOWN] Running analysis before exit…")
        _run_post_exam_analysis()

    if _exam_timer:
        _exam_timer.cancel()

    _analysis_executor.shutdown(wait=True)

    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

# ─── App Setup ────────────────────────────────────────────────────────────────

app = web.Application(middlewares=[cors_middleware])
app.router.add_route("POST",    "/offer", offer)
app.router.add_route("OPTIONS", "/offer", lambda request: web.Response())
app.on_shutdown.append(on_shutdown)

# ─── Main Runner ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import ssl

    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(
        certfile="10.100.103.0+1.pem",
        keyfile="10.100.103.0+1-key.pem"
    )

    print("=" * 60)
    print(f"  Proctoring WebRTC Server  —  https://0.0.0.0:8081")
    print(f"  Exam duration  : {EXAM_DURATION_SECONDS // 60} minutes")
    print(f"  Frames saved   : {FRAMES_DIR}/")
    print(f"  Annotated out  : {ANNOTATED_DIR}/")
    print("=" * 60)

    web.run_app(
        app,
        host="0.0.0.0",
        port=8081,
        ssl_context=ssl_context
    )