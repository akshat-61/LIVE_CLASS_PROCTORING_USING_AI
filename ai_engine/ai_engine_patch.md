# ai_engine.py — Auto-Threshold Integration Patch
# ================================================
# Apply these changes to your existing ai_engine.py.
# Sections are labelled with the line numbers from your current file.

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 — Fix broken config lines (line 179-181)
# ─────────────────────────────────────────────────────────────────────────────
# REMOVE these three lines:
#   PHONE_CONF_THRESHOLD = config.get("thresholds", "phone_conf")
#   GAZE_LEFT_THRESHOLD = config.get("thresholds", "gaze_left")
#   AI_FRAME_SIZE = tuple(config.get("system", "frame_size"))

# REPLACE WITH:
PHONE_CONF_THRESHOLD = 0.45          # was referencing undefined 'config'
GAZE_LEFT_THRESHOLD  = 0.42          # was referencing undefined 'config'
AI_FRAME_SIZE        = (960, 540)    # was referencing undefined 'config'

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2 — Import AutoThreshold (top of file, after existing imports ~line 26)
# ─────────────────────────────────────────────────────────────────────────────
from auto_threshold import AutoThreshold, tick_auto_threshold

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3 — Instantiate AutoThreshold at module level (after learner ~line 105)
# ─────────────────────────────────────────────────────────────────────────────
# ADD after:   learner.load_baselines()
auto_thr = AutoThreshold()

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 4 — Add auto_thr.reset() to reset_session() (inside reset_session ~line 272)
# ─────────────────────────────────────────────────────────────────────────────
# ADD at the end of reset_session(), before the final log.info:
auto_thr.reset()

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 5 — Replace the calibration block inside run_ai_on_frame (lines 1156-1199)
# ─────────────────────────────────────────────────────────────────────────────
# FIND this block (starts around line 1156):
#
#   if not locked:
#       stable_counter = stable_counter + 1 if len(current_ids) > 0 else 0
#       ...
#       if stable_counter >= STABILITY_FRAMES:
#           locked = True
#           ...
#       return frame
#
# REPLACE with the following:

    if not locked:
        stable_counter = stable_counter + 1 if len(current_ids) > 0 else 0

        for tid, pos in current_positions.items():
            if tid not in _calib_accumulated_positions:
                _calib_accumulated_positions[tid] = pos
            else:
                old = _calib_accumulated_positions[tid]
                _calib_accumulated_positions[tid] = (
                    int(old[0] * 0.7 + pos[0] * 0.3),
                    int(old[1] * 0.7 + pos[1] * 0.3),
                )

        # ── Legacy AdaptiveLearner calibration (keeps baseline EMA running) ──
        calib_samples = learner.on_calibration_frame(
            img_rgb, current_positions, face_mesh, pose_detector, iw, ih,
            mesh_res=mesh_res, pose_res=pose_res
        )

        # ── NEW: AutoThreshold per-student signal collection ──────────────────
        # Build a temporary student_id_map for the pre-lock phase so tick_auto_threshold
        # can map track IDs → student IDs even before IDs are finalised.
        # We use a provisional map based on sorted position (same as assign_student_ids).
        _provisional_map = {
            tid: f"S{i:03d}"
            for i, (tid, _) in enumerate(
                sorted(current_positions.items(), key=lambda kv: (kv[1][1], kv[1][0])),
                1,
            )
        }
        tick_auto_threshold(
            auto_thr, learner,
            _provisional_map,
            current_positions,
            mesh_res, pose_res,
            iw, ih,
        )

        # ── Progress bar ──────────────────────────────────────────────────────
        progress = int((stable_counter / STABILITY_FRAMES) * (iw - 40))
        cv2.rectangle(frame, (20, ih - 30), (iw - 20, ih - 10), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, ih - 30), (20 + progress, ih - 10), (0, 255, 255), -1)

        # Show per-student auto-threshold progress
        at_status = " | ".join(
            f"{sid}:{p}/{t}" for sid, (p, t) in [
                (sid, auto_thr.calibration_progress(sid))
                for sid in _provisional_map.values()
                if not auto_thr.is_ready(sid)
            ]
        ) or "auto-thr OK"

        cv2.putText(
            frame,
            f"CALIBRATING ({stable_counter}/{STABILITY_FRAMES}) | {at_status} | samples: {calib_samples}",
            (20, ih - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1,
        )

        if stable_counter >= STABILITY_FRAMES:
            locked      = True
            _lock_frame = frame_count
            all_positions = _calib_accumulated_positions
            matched_map   = learner.match_seats(all_positions)
            if matched_map:
                student_id_map.update(matched_map)
                log.info("Seat-based matching: %d matched.", len(matched_map))
            else:
                student_id_map.update(assign_student_ids(all_positions))
                log.info("Fresh IDs assigned to %d students.", len(all_positions))
            learner.on_lock(student_id_map)

            # ── Remap auto_thr from provisional → final student IDs ──────────
            # Provisional IDs and final IDs are both position-sorted S001/S002 etc,
            # so they should already match. But we re-run finalise for any remaining
            # uncompleted students using the now-final IDs.
            for tid, sid in student_id_map.items():
                prov_sid = _provisional_map.get(tid)
                if prov_sid and prov_sid != sid:
                    # Edge case: remap buffer if IDs differ
                    if prov_sid in auto_thr._buffers:
                        auto_thr._buffers[sid] = auto_thr._buffers.pop(prov_sid)
                    if prov_sid in auto_thr._finalised:
                        auto_thr._finalised.discard(prov_sid)
                        auto_thr._finalised.add(sid)
                    if prov_sid in auto_thr._thresholds:
                        auto_thr._thresholds[sid] = auto_thr._thresholds.pop(prov_sid)

                # Finalise any students not yet done
                if not auto_thr.is_ready(sid):
                    thr = auto_thr.finalise(sid)
                    bl  = learner._baselines.get(sid)
                    if bl is not None:
                        auto_thr.apply_to_learner(sid, bl)

            seat_anchor_positions = dict(_calib_accumulated_positions)
            init_seat_zones(seat_anchor_positions)
            learner.set_seat_positions(seat_positions)
            log.info("LOCKED — %d student(s): %s", len(student_id_map), list(student_id_map.values()))
            log.info("Seat zones: %d", len(seat_positions))

        return frame


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 6 — Use per-student thresholds in get_gaze_direction() (line 471-485)
# ─────────────────────────────────────────────────────────────────────────────
# The current get_gaze_direction() uses global GAZE_LEFT_THRESHOLD / GAZE_RIGHT_THRESHOLD.
# Replace with a version that accepts per-student thresholds:

def get_gaze_direction(landmarks, gaze_thr_low=None, gaze_thr_high=None):
    """
    Compute gaze ratio and determine left/right gaze.

    gaze_thr_low  : per-student lower threshold (fallback to GAZE_LEFT_THRESHOLD)
    gaze_thr_high : per-student upper threshold (fallback to GAZE_RIGHT_THRESHOLD)
    """
    left_iris_x  = landmarks[468].x
    right_iris_x = landmarks[473].x
    left_outer   = landmarks[33].x
    left_inner   = landmarks[133].x
    right_outer  = landmarks[263].x
    right_inner  = landmarks[362].x
    left_ratio  = (left_iris_x  - left_outer)  / (left_inner  - left_outer  + 1e-6)
    right_ratio = (right_iris_x - right_inner) / (right_outer - right_inner + 1e-6)
    iris_gaze = (left_ratio + right_ratio) / 2
    nose_x    = landmarks[1].x
    head_ratio = (nose_x - left_outer) / (right_outer - left_outer + 1e-6)

    thr_low  = gaze_thr_low  if gaze_thr_low  is not None else GAZE_LEFT_THRESHOLD
    thr_high = gaze_thr_high if gaze_thr_high is not None else GAZE_RIGHT_THRESHOLD

    looking_left  = (iris_gaze  < thr_low)  or (head_ratio < 0.35)
    looking_right = (iris_gaze  > thr_high) or (head_ratio > 0.65)
    return iris_gaze, looking_left, looking_right


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 7 — Pass per-student thresholds to get_gaze_direction() call site
#           (inside the face mesh loop in run_ai_on_frame, ~line 1297)
# ─────────────────────────────────────────────────────────────────────────────
# FIND:
#   gaze_ratio, looking_left, looking_right = get_gaze_direction(pts)

# REPLACE WITH:
        _gaze_thr = auto_thr.get(sid, "gaze")
        _gaze_thr_low  = _gaze_thr.thr_low  if _gaze_thr else None
        _gaze_thr_high = _gaze_thr.thr_high if _gaze_thr else None
        gaze_ratio, looking_left, looking_right = get_gaze_direction(
            pts, gaze_thr_low=_gaze_thr_low, gaze_thr_high=_gaze_thr_high
        )


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 8 — Post-lock secondary calibration (optional but recommended)
#           Lines 1554-1562 — the post-lock on_calibration_frame call
# ─────────────────────────────────────────────────────────────────────────────
# The existing code already has a post-lock calibration block for uncalibrated students.
# After the auto_threshold changes, some students may still be uncalibrated (e.g.,
# they were occluded during the window). Add auto_thr feeding here too:

# FIND (~line 1554):
#   _all_calibrated = all(
#       learner.is_calibrated(student_id_map[tid])
#       ...
#   if not _all_calibrated and frame_count % 2 == 0:
#       learner.on_calibration_frame(...)

# ADD after learner.on_calibration_frame(...) call:
        # Also feed AutoThreshold for any students not yet finalised
        for tid in current_ids:
            sid = student_id_map.get(tid)
            if sid and not auto_thr.is_ready(sid):
                if auto_thr.should_finalise(sid):
                    thr = auto_thr.finalise(sid)
                    bl  = learner._baselines.get(sid)
                    if bl is not None:
                        auto_thr.apply_to_learner(sid, bl)
