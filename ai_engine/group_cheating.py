import time
from collections import deque

SYNC_WINDOW_FRAMES   = 10
SYNC_MIN_STUDENTS    = 3
SYNC_COOLDOWN_SEC    = 30.0
REPEAT_WINDOW_FRAMES = 90
REPEAT_THRESHOLD     = 3
REPEAT_COOLDOWN_SEC  = 20.0

_frame_gaze_log: deque = deque(maxlen=SYNC_WINDOW_FRAMES)
_gaze_event_log: dict  = {}
_last_sync_alert: dict = {}
_last_repeat_alert: dict = {}


def record_frame_gazes(frame_count: int, student_gazes: dict):
    _frame_gaze_log.append((frame_count, dict(student_gazes)))


def check_synchronized_gaze(frame_count: int, can_send_fn, send_fn, score_fn):
    if len(_frame_gaze_log) < SYNC_WINDOW_FRAMES:
        return None

    for direction in ("left", "right"):
        counts_per_frame = []
        for fc, gazes in _frame_gaze_log:
            c = sum(1 for d in gazes.values() if d == direction)
            counts_per_frame.append(c)

        sustained = sum(1 for c in counts_per_frame if c >= SYNC_MIN_STUDENTS)
        if sustained >= int(SYNC_WINDOW_FRAMES * 0.6):
            key = f"sync_gaze_{direction}"
            now = time.time()
            last = _last_sync_alert.get(key, 0.0)
            if now - last > SYNC_COOLDOWN_SEC:
                _last_sync_alert[key] = now
                event = f"SYNCHRONIZED_GAZE_{direction.upper()}"
                send_fn(event, f"ROOM_{direction.upper()}")
                involved = []
                for _, gazes in list(_frame_gaze_log)[-3:]:
                    for sid, d in gazes.items():
                        if d == direction and sid not in involved:
                            involved.append(sid)
                for sid in involved:
                    score_fn(sid, "SYNCHRONIZED_GAZE")
                return event, involved

    return None


def record_gaze_event(sid: str, direction: str, frame_count: int):
    if sid not in _gaze_event_log:
        _gaze_event_log[sid] = {
            "left":  deque(maxlen=REPEAT_WINDOW_FRAMES),
            "right": deque(maxlen=REPEAT_WINDOW_FRAMES),
            "down":  deque(maxlen=REPEAT_WINDOW_FRAMES),
        }
    if direction not in _gaze_event_log[sid]:
        return None
    _gaze_event_log[sid][direction].append(frame_count)

    recent = [f for f in _gaze_event_log[sid][direction]
              if frame_count - f <= REPEAT_WINDOW_FRAMES]
    if len(recent) >= REPEAT_THRESHOLD:
        key = f"gaze_repeat_{direction}_{sid}"
        now = time.time()
        last = _last_repeat_alert.get(key, 0.0)
        if now - last > REPEAT_COOLDOWN_SEC:
            _last_repeat_alert[key] = now
            return f"REPEATED_GAZE_{direction.upper()}"
    return None


def reset():
    _frame_gaze_log.clear()
    _gaze_event_log.clear()
    _last_sync_alert.clear()
    _last_repeat_alert.clear()


def prune(active_sids: set):
    for k in list(_gaze_event_log.keys()):
        if k not in active_sids:
            del _gaze_event_log[k]
