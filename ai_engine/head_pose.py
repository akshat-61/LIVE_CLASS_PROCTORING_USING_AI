import cv2
import numpy as np
from collections import deque

_3D_MODEL_POINTS = np.array([
    [0.0,    0.0,    0.0   ],
    [0.0,   -330.0, -65.0  ],
    [-225.0, 170.0, -135.0 ],
    [225.0,  170.0, -135.0 ],
    [-150.0,-150.0, -125.0 ],
    [150.0, -150.0, -125.0 ],
], dtype=np.float64)

HEAD_YAW_THRESHOLD   = 20.0
HEAD_PITCH_THRESHOLD = 18.0
HEAD_ROLL_THRESHOLD  = 20.0

_pose_history: dict = {}
_POSE_SMOOTH = 5


def get_head_pose(landmarks, iw: int, ih: int):
    pts_2d = np.array([
        [landmarks[1].x   * iw, landmarks[1].y   * ih],
        [landmarks[152].x * iw, landmarks[152].y * ih],
        [landmarks[33].x  * iw, landmarks[33].y  * ih],
        [landmarks[263].x * iw, landmarks[263].y * ih],
        [landmarks[61].x  * iw, landmarks[61].y  * ih],
        [landmarks[291].x * iw, landmarks[291].y * ih],
    ], dtype=np.float64)

    focal      = float(iw)
    cam_matrix = np.array(
        [[focal, 0, iw / 2.0],
         [0, focal, ih / 2.0],
         [0, 0, 1.0]],
        dtype=np.float64
    )
    dist_coeff = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        _3D_MODEL_POINTS, pts_2d, cam_matrix, dist_coeff,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = angles[0]
    yaw   = angles[1]
    roll  = angles[2]
    return pitch, yaw, roll


def smooth_head_pose(sid: str, pitch: float, yaw: float, roll: float):
    if sid not in _pose_history:
        _pose_history[sid] = {
            "pitch": deque(maxlen=_POSE_SMOOTH),
            "yaw":   deque(maxlen=_POSE_SMOOTH),
            "roll":  deque(maxlen=_POSE_SMOOTH),
        }
    h = _pose_history[sid]
    h["pitch"].append(pitch)
    h["yaw"].append(yaw)
    h["roll"].append(roll)
    return (
        sum(h["pitch"]) / len(h["pitch"]),
        sum(h["yaw"])   / len(h["yaw"]),
        sum(h["roll"])  / len(h["roll"]),
    )


def classify_head_pose(pitch: float, yaw: float, roll: float):
    turned_left  = yaw  < -HEAD_YAW_THRESHOLD
    turned_right = yaw  >  HEAD_YAW_THRESHOLD
    pitched_down = pitch < -HEAD_PITCH_THRESHOLD
    tilted       = abs(roll) > HEAD_ROLL_THRESHOLD
    return turned_left, turned_right, pitched_down, tilted


def reset_pose_history():
    _pose_history.clear()


def prune_pose_history(active_sids: set):
    for k in list(_pose_history.keys()):
        if k not in active_sids:
            del _pose_history[k]
