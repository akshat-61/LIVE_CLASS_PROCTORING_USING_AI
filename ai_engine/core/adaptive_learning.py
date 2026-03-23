import json
import os
import time
import numpy as np
import cv2
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class Config:
    BASELINE_ALPHA     = 0.008
    CALIBRATION_FRAMES = 3
    COLLECT_EVERY_N    = 6

    REL_LOOK_DOWN = 0.022
    REL_YAW       = 0.045
    REL_LEAN      = 0.040 
    CALIBRATION_FRAMES_LEAN = 3
    REL_MAR = 0.040
    SEAT_DRIFT_ALPHA = 0.10
    SEAT_MATCH_THRESHOLD = 120

    FALLBACK_LOOK_DOWN = 0.042
    FALLBACK_YAW_LEFT  = 0.45  
    FALLBACK_YAW_RIGHT = 0.55  
    FALLBACK_LEAN      = 0.10  
    FALLBACK_MAR       = 0.18

    PROFILES_PATH = "student_profiles.json"
    AUTO_SAVE     = True
    DEBUG         = True
    DEBUG_MAX_ROWS = 6

    BEHAVIOR_WEIGHTS = {
        "LOOKING_LEFT": 2,
        "LOOKING_RIGHT": 2,
        "LOOKING_DOWN": 2,
        "BODY_LEANING": 3,
        "TALKING_DETECTED": 4,
        "WHISPERING_DETECTED": 6,
        "UNAUTHORIZED_OBJECT": 8,
        "PHONE_DETECTED": 10,
        "BURST_ACTIVITY": 5,
    }

    SCORE_DECAY_PER_FRAME = 0.02

    RISK_THRESHOLDS = {
        "LOW": 6,
        "HIGH": 13,
        "CRITICAL": 21,
    }

@dataclass
class StudentBaseline:
    look_down : float = 0.042
    yaw       : float = 0.500
    mar       : float = 0.015
    lean      : float = 0.15   

    sample_count      : int   = 0
    lean_sample_count : int   = 0

    last_updated : float = field(default_factory=time.time)


@dataclass
class DetectionResult:
    looking_down  : bool  = False
    looking_left  : bool  = False
    looking_right : bool  = False
    talking       : bool  = False
    is_calibrated : bool  = False

    dev_look_down : float = 0.0
    dev_yaw       : float = 0.0
    dev_mar       : float = 0.0

@dataclass
class BehaviorState:
    score: float = 0.0
    last_event_time: float = 0.0
    risk_level: str = "NORMAL"

class AdaptiveLearner:

    def __init__(self, config: Config = None):
        self.cfg = config or Config()

        self._calib_buffer : Dict[int, Dict[str, List[float]]] = {}
        self._baselines    : Dict[str, StudentBaseline]        = {}

        self._locked        = False
        self._student_map   : Dict[int, str] = {}
        self._frame_count   = 0
        self._session_start = time.time()
        self._last_results  : Dict[str, DetectionResult] = {}
        self._behavior: Dict[str, BehaviorState] = {}
        self._event_history: Dict[str, List[Tuple[str, float]]] = {}
        self._seat_positions: Dict[str, Tuple[int, int]] = {}
        self.SEQUENCE_WINDOW = 5.0
        self.SEQUENCE_BONUS = 6
        self._stability_counters: Dict[str, Dict[str, int]] = {}
        self._coordination_last_trigger = {}
        self.COORDINATION_COOLDOWN = 5.0
        self.STABILITY_FRAMES = 4
        self._burst_last_trigger = {}
        self.BURST_COOLDOWN = 4.0
        
        self._session_events = []
        self._session_start_time = time.time()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._session_filename = f"SESSION_{timestamp}.json"

        os.makedirs("session_logs", exist_ok=True)
        
        print("[AdaptiveLearner] Initialised — Phase 1 ready")
        print(f"  EMA alpha:            {self.cfg.BASELINE_ALPHA}")
        print(f"  Calibration frames:   {self.cfg.CALIBRATION_FRAMES}")
        print(f"  Lean calib frames:    {self.cfg.CALIBRATION_FRAMES_LEAN}")
        print(f"  Rel lean threshold:   {self.cfg.REL_LEAN}")
        print(f"  Fallback lean:        {self.cfg.FALLBACK_LEAN}")

    def on_calibration_frame(
        self,
        img_rgb,
        current_positions: Dict[int, Tuple[int, int]],
        face_mesh,
        pose_detector,
        iw: int,
        ih: int,
        mesh_res=None,
        pose_res=None,
    ) -> int:
        self._frame_count += 1
        if self._frame_count % self.cfg.COLLECT_EVERY_N != 0:
            return self._total_calib_samples()

        
        try:
            if mesh_res is None:
                mesh_res = face_mesh.process(img_rgb)
            if mesh_res and mesh_res.multi_face_landmarks:
                for f_lms in mesh_res.multi_face_landmarks:
                    pts = f_lms.landmark
                    tid = self._get_student_at(
                        int(pts[1].x * iw), int(pts[1].y * ih), current_positions
                    )
                    if tid is None:
                        continue
                    buf = self._get_calib_buf(tid)
                    buf["look_down"].append(self._get_look_down_score(pts))
                    left_iris   = pts[468].x
                    right_iris  = pts[473].x
                    left_outer  = pts[33].x    
                    left_inner  = pts[133].x   
                    right_outer = pts[263].x   
                    right_inner = pts[362].x   
                    left_ratio  = (left_iris  - left_outer)  / (left_inner  - left_outer  + 1e-6)
                    
                    right_ratio = (right_iris - right_inner) / (right_outer - right_inner + 1e-6)
                    gaze_ratio  = (left_ratio + right_ratio) / 2
                    buf["yaw"].append(gaze_ratio)
                    buf["mar"].append(self._mouth_aspect_ratio(pts))
        except Exception as e:
            print(f"[AdaptiveLearner] Face mesh calibration error: {e}")

        try:
            if pose_res is None:
                pose_res = pose_detector.process(img_rgb)
            if pose_res and pose_res.pose_landmarks:
                lms = pose_res.pose_landmarks.landmark
                tid = self._get_student_at(
                    int(lms[0].x * iw), int(lms[0].y * ih), current_positions
                )
                if tid is not None:
                    buf = self._get_calib_buf(tid)
                    buf["lean"].append(abs(lms[11].x - lms[12].x))
        except Exception as e:
            print(f"[AdaptiveLearner] Pose calibration error: {e}")

        return self._total_calib_samples()
    
    def match_seats(self, current_positions: dict) -> dict:
        if not self._seat_positions:
            return {}

        new_map = {}
        used_saved_ids = set()

        for tid, pos in current_positions.items():
            best_sid = None
            best_dist = 999999

            for saved_sid, saved_pos in self._seat_positions.items():
                if saved_sid in used_saved_ids:
                    continue

                dx = pos[0] - saved_pos[0]
                dy = pos[1] - saved_pos[1]
                d = (dx * dx + dy * dy) ** 0.5

                if d < best_dist:
                    best_dist = d
                    best_sid = saved_sid

            if best_sid and best_dist < self.cfg.SEAT_MATCH_THRESHOLD:
                new_map[tid] = best_sid
                used_saved_ids.add(best_sid)

                old_pos = self._seat_positions[best_sid]
                alpha = self.cfg.SEAT_DRIFT_ALPHA

                new_x = int(alpha * pos[0] + (1 - alpha) * old_pos[0])
                new_y = int(alpha * pos[1] + (1 - alpha) * old_pos[1])

                self._seat_positions[best_sid] = (new_x, new_y)

        return new_map

    def on_lock(self, student_id_map: Dict[int, str]):
        self._student_map = student_id_map
        self._locked      = True

        transferred = 0
        for tid, measurements in self._calib_buffer.items():
            sid = student_id_map.get(tid)
            if sid is None:
                continue

            bl = StudentBaseline()

            for key in ["look_down", "yaw", "mar"]:
                values = measurements.get(key, [])
                if values:
                    setattr(bl, key, sum(values) / len(values))

            lean_values = measurements.get("lean", [])
            if lean_values:
                bl.lean = sum(lean_values) / len(lean_values)
                bl.lean_sample_count = len(lean_values)
            else:
                bl.lean_sample_count = 0

            face_counts = [
                len(measurements.get(k, []))
                for k in ["look_down", "yaw", "mar"]
            ]
            bl.sample_count = min(face_counts) if all(c > 0 for c in face_counts) else 0
            bl.last_updated = time.time()

            self._baselines[sid] = bl
            transferred += 1

        self._calib_buffer.clear()

        calibrated_baselines = [
            bl for bl in self._baselines.values()
            if bl.sample_count >= self.cfg.CALIBRATION_FRAMES
        ]
        if calibrated_baselines:
            median_look_down = float(np.median([b.look_down for b in calibrated_baselines]))
            median_yaw       = float(np.median([b.yaw       for b in calibrated_baselines]))
            median_mar       = float(np.median([b.mar       for b in calibrated_baselines]))
        else:
            median_look_down = self.cfg.FALLBACK_LOOK_DOWN
            median_yaw       = (self.cfg.FALLBACK_YAW_LEFT + self.cfg.FALLBACK_YAW_RIGHT) / 2
            median_mar       = self.cfg.FALLBACK_MAR

        for tid, sid in student_id_map.items():
            if sid not in self._baselines:
                bl = StudentBaseline()
                # Mark as calibrated so detections fire even on fallback values
                bl.sample_count      = self.cfg.CALIBRATION_FRAMES
                bl.lean_sample_count = self.cfg.CALIBRATION_FRAMES_LEAN
                self._baselines[sid] = bl
                print(f"  [AdaptiveLearner] ⚠ No calib data for {sid} — using defaults (auto-marked calibrated)")
            else:
                bl = self._baselines[sid]
                if bl.sample_count < self.cfg.CALIBRATION_FRAMES:
                    if bl.sample_count == 0:
                        bl.look_down = median_look_down
                        bl.yaw       = median_yaw
                        bl.mar       = median_mar
                        # Mark as calibrated so detections fire
                        bl.sample_count = self.cfg.CALIBRATION_FRAMES
                    print(f"  [AdaptiveLearner] ⚠ {sid} face incomplete "
                          f"— patched with classroom median (auto-marked calibrated)")

        print(f"\n[AdaptiveLearner] ✅ Locked — baselines for {transferred} students:")
        for sid, bl in self._baselines.items():
            face_ok  = "✅" if self.is_calibrated(sid)      else f"⏳ {bl.sample_count}/{self.cfg.CALIBRATION_FRAMES}"
            lean_ok  = "✅" if self.is_lean_calibrated(sid) else f"⏳ {bl.lean_sample_count}/{self.cfg.CALIBRATION_FRAMES_LEAN}"
            print(f"  {sid}: ld={bl.look_down:.4f} yw={bl.yaw:.4f} "
                  f"mr={bl.mar:.4f} ln={bl.lean:.4f} "
                  f"[face:{face_ok}] [lean:{lean_ok}]")
        print()
    
    

    def on_detection_frame(
        self,
        sid: str,
        look_down_raw: float,
        gaze_ratio: float,
        mar_smoothed: float,
        mar_variation: float,
        mouth_open_vertical: bool,
    ):
        result               = DetectionResult()
        result.is_calibrated = self.is_calibrated(sid)

        bl = self._baselines.get(sid)
        if bl is None:
            self._last_results[sid] = result
            return result

        if look_down_raw < bl.look_down + self.cfg.REL_LOOK_DOWN * 2:
            self._update_ema(bl, "look_down", look_down_raw)
        if abs(gaze_ratio - bl.yaw) < self.cfg.REL_YAW * 2:
            self._update_ema(bl, "yaw", gaze_ratio)
        if mar_smoothed < 0.10:
            self._update_ema(bl, "mar", mar_smoothed)

        dev_down = look_down_raw - bl.look_down
        dev_yaw  = gaze_ratio - bl.yaw
        dev_mar  = mar_smoothed  - bl.mar

        result.dev_look_down = dev_down
        result.dev_yaw       = dev_yaw
        result.dev_mar       = dev_mar

        if result.is_calibrated:
            result.looking_down  = dev_down >  self.cfg.REL_LOOK_DOWN
            result.looking_left  = dev_yaw  < -self.cfg.REL_YAW
            result.looking_right = dev_yaw  >  self.cfg.REL_YAW
            result.talking       = (dev_mar > self.cfg.REL_MAR
                                    and mar_smoothed < 0.90
                                    and mar_variation > 0.008
                                    and mouth_open_vertical)
        else:
            result.looking_down  = False
            result.looking_left  = False
            result.looking_right = False
            result.talking       = (dev_mar > self.cfg.REL_MAR
                                    and mar_smoothed < 0.90
                                    and mar_variation > 0.008
                                    and mouth_open_vertical)

        self._last_results[sid] = result
        return result

    def check_lean(self, sid: str, shoulder_diff_x: float) -> bool:
        
        bl = self._baselines.get(sid)
        if bl is None:
            return shoulder_diff_x < self.cfg.FALLBACK_LEAN

        alpha = 0.05 if not self.is_lean_calibrated(sid) else self.cfg.BASELINE_ALPHA
        old = bl.lean
        bl.lean = alpha * shoulder_diff_x + (1 - alpha) * old
        bl.lean_sample_count += 1
        bl.last_updated = time.time()

        if self.is_lean_calibrated(sid):
            return abs(shoulder_diff_x - bl.lean) > self.cfg.REL_LEAN
        else:
            return shoulder_diff_x < self.cfg.FALLBACK_LEAN

    def is_calibrated(self, sid: str) -> bool:
        bl = self._baselines.get(sid)
        return bl is not None and bl.sample_count >= self.cfg.CALIBRATION_FRAMES

    def is_lean_calibrated(self, sid: str) -> bool:
        bl = self._baselines.get(sid)
        return bl is not None and bl.lean_sample_count >= self.cfg.CALIBRATION_FRAMES_LEAN

    def calibration_progress(self, sid: str) -> Tuple[int, int]:
        bl = self._baselines.get(sid)
        return (bl.sample_count if bl else 0), self.cfg.CALIBRATION_FRAMES

    def lean_calibration_progress(self, sid: str) -> Tuple[int, int]:
        bl = self._baselines.get(sid)
        return (bl.lean_sample_count if bl else 0), self.cfg.CALIBRATION_FRAMES_LEAN

    def get_baseline_dict(self, sid: str) -> dict:
        bl = self._baselines.get(sid)
        return asdict(bl) if bl else {}

    def total_calib_samples(self) -> int:
        return self._total_calib_samples()
    
    def register_event(self, sid: str, event_type: str):
        if sid not in self._behavior:
            self._behavior[sid] = BehaviorState()

        state = self._behavior[sid]
        now = time.time()
        if sid not in self._event_history:
            self._event_history[sid] = []
        
        self._event_history[sid].append((event_type, now))
        
        self._event_history[sid] = [
            (e, t) for (e, t) in self._event_history[sid]
            if now - t <= self.SEQUENCE_WINDOW
        ]

        self._session_events.append({
            "timestamp": time.time(),
            "student_id": sid,
            "event_type": event_type,
            "risk_score": state.score,
            "risk_level": state.risk_level
        })

        weight = self.cfg.BEHAVIOR_WEIGHTS.get(event_type, 0)
        if weight == 0:
            return

        now = time.time()

        if now - state.last_event_time > 1.5:
            state.score += weight
            state.last_event_time = now
        history = [e for (e, t) in self._event_history[sid]]
        recent_events = [
            (e, t) for (e, t) in self._event_history[sid]
            if time.time() - t <= self.SEQUENCE_WINDOW
        ]
        
        if len(recent_events) >= 3:
            now = time.time()
            last = self._burst_last_trigger.get(sid, 0)

            if now - last >= self.BURST_COOLDOWN:
                print(f"[Burst Detected] Rapid suspicious activity by {sid}")

                burst_weight = self.cfg.BEHAVIOR_WEIGHTS.get("BURST_ACTIVITY", 4)
                state.score += burst_weight

                self._session_events.append({
                    "timestamp": time.time(),
                    "student_id": sid,
                    "event_type": "BURST_ACTIVITY",
                    "risk_score": state.score,
                    "risk_level": state.risk_level
                })

                self._burst_last_trigger[sid] = now

        if self._detect_sequence(
                history,
                ["LOOKING_LEFT", "BODY_LEANING", "LOOKING_RIGHT"]):

            print(f"[Sequence Detected] Coordinated movement by {sid}")
            state.score += self.SEQUENCE_BONUS
            self._event_history[sid].clear()

        self._update_risk_level(state)

    def decay_scores(self):
        for state in self._behavior.values():
            state.score = max(0.0, state.score - self.cfg.SCORE_DECAY_PER_FRAME)
            self._update_risk_level(state)

    def _build_session_summary(self):
        summary = {}

        for sid, state in self._behavior.items():
            student_events = [e for e in self._session_events if e["student_id"] == sid]

            summary[sid] = {
                "total_events": len(student_events),
                "final_risk_score": round(state.score, 2),
                "final_risk_level": state.risk_level
            }

        return summary

    def get_behavior_state(self, sid: str) -> BehaviorState:
        return self._behavior.get(sid, BehaviorState())
    
    def escalate_if_coordinated(self, sid_a: str, sid_b: str):
        state_a = self._behavior.get(sid_a)
        state_b = self._behavior.get(sid_b)

        if not state_a or not state_b:
            return

        if state_a.risk_level != "HIGH" or state_b.risk_level != "HIGH":
            return

        pair_key = tuple(sorted([sid_a, sid_b]))
        now = time.time()

        last_time = self._coordination_last_trigger.get(pair_key, 0)

        if now - last_time < self.COORDINATION_COOLDOWN:
            return  

        print(f"[Coordination Detected] {sid_a} & {sid_b}")

        state_a.score += 5
        state_b.score += 5

        self._update_risk_level(state_a)
        self._update_risk_level(state_b)

        self._coordination_last_trigger[pair_key] = now

    def _update_risk_level(self, state: BehaviorState):
        score = state.score
        thresholds = self.cfg.RISK_THRESHOLDS

        if score >= thresholds["CRITICAL"]:
            state.risk_level = "CRITICAL"
        elif score >= thresholds["HIGH"]:
            state.risk_level = "HIGH"
        elif score >= thresholds["LOW"]:
            state.risk_level = "LOW"
        else:
            state.risk_level = "NORMAL"
    
    def _detect_sequence(self, history: List[str], pattern: List[str]) -> bool:
        
        if len(history) < len(pattern):
            return False

        idx = 0
        for event in history:
            if event == pattern[idx]:
                idx += 1
                if idx == len(pattern):
                    return True

        return False
    
    def _stable_trigger(self, sid: str, event_name: str, condition: bool) -> bool:

        if not hasattr(self, "_stability_counters"):
            self._stability_counters = {}
    
        if sid not in self._stability_counters:
            self._stability_counters[sid] = {}
    
        counter = self._stability_counters[sid].get(event_name, 0)
    
        if condition:
            counter += 1
        else:
            counter = 0
    
        self._stability_counters[sid][event_name] = counter
    
        return counter >= self.STABILITY_FRAMES

    def draw_debug(self, frame, iw: int):
        if not self.cfg.DEBUG:
            return
        if not self._baselines and not self._calib_buffer:
            return

        x_start = iw - 240
        y       = 55
        font    = cv2.FONT_HERSHEY_SIMPLEX

        cv2.rectangle(frame, (x_start - 4, y - 14), (iw - 4, y + 4), (20, 20, 30), -1)
        cv2.putText(frame, "ADAPTIVE BASELINES", (x_start, y), font, 0.33, (0, 200, 255), 1)
        y += 14

        shown = 0
        for sid, bl in list(self._baselines.items()):
            if shown >= self.cfg.DEBUG_MAX_ROWS:
                break

            face_ok = self.is_calibrated(sid)
            lean_ok = self.is_lean_calibrated(sid)
            lp, lt  = self.lean_calibration_progress(sid)
            fp, ft  = self.calibration_progress(sid)

            face_str = "F:OK" if face_ok else f"F:{fp}/{ft}"
            lean_str = "L:OK" if lean_ok else f"L:{lp}/{lt}"
            color    = (80, 255, 120) if (face_ok and lean_ok) else (255, 200, 50)

            cv2.rectangle(frame, (x_start - 4, y - 10), (iw - 4, y + 26), (15, 20, 30), -1)

            cv2.putText(frame, f"{sid} [{face_str}] [{lean_str}]",
                        (x_start, y), font, 0.30, color, 1)
            y += 11

            cv2.putText(frame,
                        f"  dn:{bl.look_down:.3f}  yw:{bl.yaw:.3f}  "
                        f"mr:{bl.mar:.3f}  ln:{bl.lean:.3f}",
                        (x_start, y), font, 0.28, (150, 180, 200), 1)
            y += 11

            behavior = self._behavior.get(sid)
            if behavior:
                if behavior.risk_level == "CRITICAL":
                    risk_color = (0, 0, 255)       
                elif behavior.risk_level == "HIGH":
                    risk_color = (0, 165, 255)     
                elif behavior.risk_level == "LOW":
                    risk_color = (0, 255, 255)     
                else:
                    risk_color = (0, 255, 0)       

                cv2.putText(frame,
                            f"  RISK:{behavior.risk_level}  SCORE:{behavior.score:.1f}",
                            (x_start, y),
                            font, 0.28, risk_color, 1)
                y += 11

            res = self._last_results.get(sid)
            if res and face_ok:
                cv2.putText(frame,
                            f"  dv dn:{res.dev_look_down:+.3f}  "
                            f"yw:{res.dev_yaw:+.3f}  mr:{res.dev_mar:+.3f}",
                            (x_start, y), font, 0.27, (100, 130, 160), 1)
                y += 10

            y += 3
            shown += 1
    
    def set_seat_positions(self, seat_positions: dict):
        self._seat_positions = seat_positions.copy()


    def save_baselines(self, path: str = None):
        path = path or self.cfg.PROFILES_PATH
        data = {
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "phase": 2,
            "baselines": {sid: asdict(bl) for sid, bl in self._baselines.items()},
            "seat_positions": self._seat_positions
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[AdaptiveLearner] Baselines saved → {path}")

    def load_baselines(self, path: str = None):
        path = path or self.cfg.PROFILES_PATH
        if not os.path.exists(path):
            print(f"[AdaptiveLearner] No saved baselines found at {path}")
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._seat_positions = data.get("seat_positions", {})
            loaded = 0
            for sid, bl_dict in data.get("baselines", {}).items():
                bl = StudentBaseline(**{
                    k: v for k, v in bl_dict.items()
                    if k in StudentBaseline.__dataclass_fields__
                })
                bl.sample_count      = self.cfg.CALIBRATION_FRAMES
                bl.lean_sample_count = max(bl.lean_sample_count,
                                           self.cfg.CALIBRATION_FRAMES_LEAN)
                self._baselines[sid] = bl
                loaded += 1
            print(f"[AdaptiveLearner] Loaded {loaded} saved baselines from {path}")
        except Exception as e:
            print(f"[AdaptiveLearner] ⚠ Failed to load baselines: {e}")

    def shutdown(self):
        duration = time.time() - self._session_start
        print(f"\n[AdaptiveLearner] Session ended — {duration/60:.1f} minutes")
        if self.cfg.AUTO_SAVE and self._baselines:
            self.save_baselines()
        print("[AdaptiveLearner] Final baselines:")
        for sid, bl in self._baselines.items():
            face_str = "calibrated" if self.is_calibrated(sid) else "INCOMPLETE"
            lean_str = "calibrated" if self.is_lean_calibrated(sid) else "INCOMPLETE"
            print(f"  {sid} [face:{face_str}] [lean:{lean_str}]: "
                  f"ld={bl.look_down:.4f}  yw={bl.yaw:.4f}  "
                  f"mr={bl.mar:.4f}  ln={bl.lean:.4f}")
        session_duration = round(time.time() - self._session_start_time, 2)

        session_data = {
            "session_start": self._session_start_time,
            "session_end": time.time(),
            "duration_seconds": session_duration,
            "total_events": len(self._session_events),
            "events": self._session_events,
            "summary": self._build_session_summary()
        }

        log_path = os.path.join("session_logs", self._session_filename)

        with open(log_path, "w") as f:
            json.dump(session_data, f, indent=4)

        print(f"[Session Log Saved] → {log_path}")

    def _get_calib_buf(self, tid: int) -> Dict[str, List[float]]:
        if tid not in self._calib_buffer:
            self._calib_buffer[tid] = {"look_down": [], "yaw": [], "mar": [], "lean": []}
        return self._calib_buffer[tid]

    def _total_calib_samples(self) -> int:
        return sum(len(b.get("look_down", [])) for b in self._calib_buffer.values())

    def _update_ema(self, bl: StudentBaseline, key: str, value: float):
        old = getattr(bl, key)
        setattr(bl, key, self.cfg.BASELINE_ALPHA * value + (1 - self.cfg.BASELINE_ALPHA) * old)
        bl.sample_count += 1
        bl.last_updated  = time.time()

    @staticmethod
    def _get_student_at(
        x: int, y: int,
        current_positions: Dict[int, Tuple[int, int]],
        radius: int = 220,
    ) -> Optional[int]:
        best_tid, best_d = None, radius
        for tid, pos in current_positions.items():
            d = float(np.linalg.norm(np.array((x, y)) - np.array(pos)))
            if d < best_d:
                best_d, best_tid = d, tid
        return best_tid

    @staticmethod
    def _get_look_down_score(landmarks) -> float:
        def pt(i):
            lm = landmarks[i]; return (lm.x, lm.y)
        li, ri = pt(468), pt(473)
        ley = (pt(159)[1] + pt(145)[1]) / 2
        rey = (pt(386)[1] + pt(374)[1]) / 2
        return ((li[1] - ley) + (ri[1] - rey)) / 2

    @staticmethod
    def _mouth_aspect_ratio(landmarks) -> float:
        def pt(i):
            lm = landmarks[i]; return np.array([lm.x, lm.y])
        v1 = np.linalg.norm(pt(13) - pt(14))
        v2 = np.linalg.norm(pt(312) - pt(317))
        return float(((v1 + v2) / 2.0) / (np.linalg.norm(pt(61) - pt(291)) + 1e-6))