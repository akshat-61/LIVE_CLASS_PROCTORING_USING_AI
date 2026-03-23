"""
intelligence_layer.py  —  High-accuracy suspicious signal detection.

Detects four signals that require sustained evidence before alerting:

  1. EYE_MOVEMENT   — iris gaze shifted for > N consecutive frames
  2. FACE_ABSENT    — student's face not detected for > N frames
  3. MULTIPLE_PERSONS — extra person in frame not matching any seat
  4. PHONE_DETECTED — phone visible for > N frames near a student

All signals feed a per-student suspicion_score dict (0–100).
The layer is stateless across calls: all state lives in IntelligenceLayer.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Dict, Optional, Set, Tuple

log = logging.getLogger(__name__)

# ── Load thresholds from config (fall back to safe defaults) ─────────────────
try:
    from config.config import cfg
    _GAZE_FRAMES   = int(cfg.intelligence.gaze_sustained_frames)
    _GAZE_CD       = float(cfg.intelligence.gaze_cooldown)
    _MULTI_CD      = float(cfg.intelligence.multi_person_cooldown)
    _PHONE_FRAMES  = int(cfg.intelligence.phone_persist_frames)
    _PHONE_CD      = float(cfg.intelligence.phone_persist_cooldown)
    _FACE_FRAMES   = int(cfg.alerts.face_absent_threshold)
    _FACE_CD       = float(cfg.alerts.face_absent_cooldown)
    _GAZE_L        = float(cfg.gaze.left_threshold)
    _GAZE_R        = float(cfg.gaze.right_threshold)
    _HEAD_L        = float(cfg.gaze.head_left)
    _HEAD_R        = float(cfg.gaze.head_right)
except Exception:
    _GAZE_FRAMES, _GAZE_CD   = 30,  10.0
    _MULTI_CD                 = 15.0
    _PHONE_FRAMES, _PHONE_CD = 45,  20.0
    _FACE_FRAMES,  _FACE_CD  = 30,  20.0
    _GAZE_L, _GAZE_R         = 0.42, 0.58
    _HEAD_L, _HEAD_R         = 0.38, 0.62


# ─────────────────────────────────────────────────────────────────────────────

class IntelligenceLayer:
    """
    Stateful per-student tracker for sustained suspicious signals.
    Instantiate once and call update() every frame.
    """

    def __init__(self) -> None:
        # Gaze: consecutive frames the iris has been shifted
        self._gaze_left_streak:  Dict[str, int] = defaultdict(int)
        self._gaze_right_streak: Dict[str, int] = defaultdict(int)

        # Face absence: consecutive frames without face detection
        self._face_absent_frames: Dict[str, int] = defaultdict(int)

        # Phone: consecutive frames phone detected near student
        self._phone_frames: Dict[str, int] = defaultdict(int)

        # Alert cooldowns  {key -> last_sent_timestamp}
        self._last_sent: Dict[str, float] = {}

        # Suspicion scores: 0–100, decays between frames
        self._suspicion: Dict[str, float] = defaultdict(float)

        log.debug("IntelligenceLayer initialised")

    # ── Public entry point ────────────────────────────────────────────────────

    def update(
        self,
        *,
        frame_count:       int,
        locked:            bool,
        current_positions: Dict[int, Tuple[int, int]],   # tid -> (x, y)
        student_id_map:    Dict[int, str],                # tid -> sid
        seat_positions:    Dict[str, Tuple[int, int]],    # sid -> (x, y)
        invigilator_tids:  Set[int],
        face_landmarks:    list,                          # mediapipe FaceMesh results
        gaze_ratios:       Dict[str, float],              # sid -> iris_gaze
        head_ratios:       Dict[str, float],              # sid -> head_yaw_ratio
        phones:            list,                          # [(cx, cy), ...]
        iw: int, ih: int,
        can_send_fn,                                      # callable(key, cooldown) -> bool
        log_event_fn,                                     # callable(student_id, event, ...) -> None
        send_async_fn,                                    # callable(event, sid) -> None
    ) -> Dict[str, list]:
        """
        Process all intelligence signals for one frame.
        Returns dict of {sid: [event_name, ...]} for any alerts fired.
        """
        if not locked:
            return {}

        fired: Dict[str, list] = defaultdict(list)
        now = time.time()

        active_sids = {student_id_map[t] for t in current_positions
                       if t not in invigilator_tids and student_id_map.get(t)}

        # 1. Eye movement — sustained gaze
        self._check_eye_movement(active_sids, gaze_ratios, head_ratios,
                                  can_send_fn, log_event_fn, send_async_fn,
                                  frame_count, fired)

        # 2. Face absence
        self._check_face_absence(seat_positions, active_sids, student_id_map,
                                  invigilator_tids, current_positions, face_landmarks,
                                  iw, ih, can_send_fn, log_event_fn, send_async_fn,
                                  frame_count, fired)

        # 3. Multiple persons
        self._check_multiple_persons(seat_positions, current_positions,
                                     student_id_map, invigilator_tids,
                                     can_send_fn, log_event_fn, send_async_fn,
                                     frame_count, fired)

        # 4. Phone persistence per student
        self._check_phone_persistence(active_sids, phones, current_positions,
                                       student_id_map, invigilator_tids,
                                       can_send_fn, log_event_fn, send_async_fn,
                                       frame_count, fired)

        # Decay suspicion scores slowly each frame
        for sid in list(self._suspicion.keys()):
            self._suspicion[sid] = max(0.0, self._suspicion[sid] * 0.997)

        return dict(fired)

    # ── Signal 1: sustained eye movement ─────────────────────────────────────

    def _check_eye_movement(
        self, active_sids, gaze_ratios, head_ratios,
        can_send_fn, log_event_fn, send_async_fn, frame_count, fired,
    ) -> None:
        for sid in active_sids:
            gaze = gaze_ratios.get(sid)
            head = head_ratios.get(sid)
            if gaze is None:
                continue

            iris_left  = gaze < _GAZE_L
            iris_right = gaze > _GAZE_R
            head_left  = (head is not None) and head < _HEAD_L
            head_right = (head is not None) and head > _HEAD_R

            looking_left  = iris_left  or head_left
            looking_right = iris_right or head_right

            if looking_left:
                self._gaze_left_streak[sid]  += 1
                self._gaze_right_streak[sid] = 0
            elif looking_right:
                self._gaze_right_streak[sid] += 1
                self._gaze_left_streak[sid]  = 0
            else:
                self._gaze_left_streak[sid]  = max(0, self._gaze_left_streak[sid]  - 1)
                self._gaze_right_streak[sid] = max(0, self._gaze_right_streak[sid] - 1)

            for direction, streak in (("LEFT",  self._gaze_left_streak[sid]),
                                       ("RIGHT", self._gaze_right_streak[sid])):
                if streak >= _GAZE_FRAMES:
                    key = f"intel_gaze_{sid}_{direction}"
                    if can_send_fn(key, _GAZE_CD):
                        event = "EYE_MOVEMENT"
                        log_event_fn(student_id=sid, event=event,
                                     confidence=round(min(streak / _GAZE_FRAMES, 1.0), 2),
                                     frame=frame_count)
                        send_async_fn(event, sid)
                        self._suspicion[sid] = min(100.0, self._suspicion[sid] + 8)
                        fired[sid].append(event)
                        log.info("[Intel] EYE_MOVEMENT %s → %s  streak=%d", sid, direction, streak)

    # ── Signal 2: face absence ────────────────────────────────────────────────

    def _check_face_absence(
        self, seat_positions, active_sids, student_id_map, invigilator_tids,
        current_positions, face_landmarks, iw, ih,
        can_send_fn, log_event_fn, send_async_fn, frame_count, fired,
    ) -> None:
        # Collect face center positions from mediapipe
        face_centers = set()
        for fl in (face_landmarks or []):
            lm = fl.landmark[1]
            face_centers.add((int(lm.x * iw), int(lm.y * ih)))

        for sid, seat_pos in seat_positions.items():
            if sid == "S???":
                continue
            # Student is considered "seen" if they are in active_sids AND
            # a face landmark center is within 80px of their seat position
            student_present = sid in active_sids
            face_near_seat = any(
                ((fc[0] - seat_pos[0])**2 + (fc[1] - seat_pos[1])**2) < 80**2
                for fc in face_centers
            )

            if student_present and face_near_seat:
                self._face_absent_frames[sid] = 0
            else:
                self._face_absent_frames[sid] += 1

            if self._face_absent_frames[sid] >= _FACE_FRAMES:
                key = f"intel_face_absent_{sid}"
                if can_send_fn(key, _FACE_CD):
                    event = "FACE_ABSENT_DISABLED"
                    log_event_fn(student_id=sid, event=event,
                                 confidence=1.0, frame=frame_count)
                    send_async_fn(event, sid)
                    self._suspicion[sid] = min(100.0, self._suspicion[sid] + 12)
                    fired[sid].append(event)
                    log.info("[Intel] FACE_ABSENT %s  frames=%d", sid,
                             self._face_absent_frames[sid])

    # ── Signal 3: multiple persons ────────────────────────────────────────────

    def _check_multiple_persons(
        self, seat_positions, current_positions, student_id_map,
        invigilator_tids, can_send_fn, log_event_fn, send_async_fn,
        frame_count, fired,
    ) -> None:
        # Any tracked person NOT in student_id_map and NOT an invigilator
        # and NOT near any known seat is an unknown intruder
        known_tids = set(student_id_map.keys())
        unknown_tids = [
            tid for tid in current_positions
            if tid not in known_tids and tid not in invigilator_tids
        ]
        if not unknown_tids:
            return

        for uid in unknown_tids:
            upos = current_positions[uid]
            near_seat = any(
                ((upos[0] - sp[0])**2 + (upos[1] - sp[1])**2) < 100**2
                for sp in seat_positions.values()
            )
            if near_seat:
                continue  # close to a known seat — likely mis-assignment, skip
            key = f"intel_multi_{uid}"
            if can_send_fn(key, _MULTI_CD):
                event = "MULTIPLE_PERSONS"
                log_event_fn(student_id="ROOM", event=event,
                             confidence=0.8, frame=frame_count)
                send_async_fn(event, "room")
                fired["ROOM"].append(event)
                log.info("[Intel] MULTIPLE_PERSONS — unknown tid=%d at %s", uid, upos)

    # ── Signal 4: phone persistence ───────────────────────────────────────────

    def _check_phone_persistence(
        self, active_sids, phones, current_positions, student_id_map,
        invigilator_tids, can_send_fn, log_event_fn, send_async_fn,
        frame_count, fired,
    ) -> None:
        # phones is a list of (cx, cy) positions from COCO detection
        phone_sids: Set[str] = set()
        for px, py in phones:
            # Assign phone to nearest non-invigilator student within 200px
            best_sid, best_d = None, 200
            for tid, pos in current_positions.items():
                if tid in invigilator_tids:
                    continue
                d = ((px - pos[0])**2 + (py - pos[1])**2) ** 0.5
                if d < best_d:
                    best_d = d
                    best_sid = student_id_map.get(tid)
            if best_sid:
                phone_sids.add(best_sid)

        for sid in active_sids:
            if sid in phone_sids:
                self._phone_frames[sid] += 1
            else:
                self._phone_frames[sid] = max(0, self._phone_frames[sid] - 1)

            if self._phone_frames[sid] >= _PHONE_FRAMES:
                key = f"intel_phone_persist_{sid}"
                if can_send_fn(key, _PHONE_CD):
                    event = "PHONE_DETECTED"
                    log_event_fn(student_id=sid, event=event,
                                 confidence=round(
                                     min(self._phone_frames[sid] / _PHONE_FRAMES, 1.0), 2),
                                 frame=frame_count)
                    send_async_fn(event, sid)
                    self._suspicion[sid] = min(100.0, self._suspicion[sid] + 20)
                    fired[sid].append(event)
                    log.info("[Intel] PHONE_DETECTED (persistent) %s  frames=%d",
                             sid, self._phone_frames[sid])

    # ── Accessors ─────────────────────────────────────────────────────────────

    def get_suspicion_score(self, sid: str) -> float:
        return round(self._suspicion.get(sid, 0.0), 1)

    def get_all_suspicion_scores(self) -> Dict[str, float]:
        return {sid: round(v, 1) for sid, v in self._suspicion.items() if v > 0}

    def reset(self) -> None:
        self._gaze_left_streak.clear()
        self._gaze_right_streak.clear()
        self._face_absent_frames.clear()
        self._phone_frames.clear()
        self._last_sent.clear()
        self._suspicion.clear()
        log.debug("IntelligenceLayer reset")
