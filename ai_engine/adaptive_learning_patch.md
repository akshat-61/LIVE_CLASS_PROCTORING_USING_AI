# adaptive_learning.py — Auto-Threshold Integration Patch
# =========================================================
# Only the sections that need to change are listed here.

# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 — Config class: add per-student REL_* override support (line 10)
# ─────────────────────────────────────────────────────────────────────────────
# The Config class currently has global REL_* values.
# We add per-student overrides stored in a dict so auto_threshold can inject
# tighter/looser values for each student without changing the global defaults.
#
# ADD these two attributes to the Config class body:

    # Per-student threshold overrides injected by AutoThreshold
    # { sid: { "REL_LOOK_DOWN": float, "REL_YAW": float, ... } }
    STUDENT_REL_OVERRIDES: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2 — on_detection_frame(): use per-student overrides (line 297)
# ─────────────────────────────────────────────────────────────────────────────
# FIND the on_detection_frame method body (starts ~line 306):
#
#   if result.is_calibrated:
#       result.looking_down  = dev_down >  self.cfg.REL_LOOK_DOWN
#       result.looking_left  = dev_yaw  < -self.cfg.REL_YAW
#       result.looking_right = dev_yaw  >  self.cfg.REL_YAW
#       result.talking       = ...
#
# REPLACE with:

        # ── Resolve per-student or global REL_* thresholds ─────────────────────
        _overrides     = self.cfg.STUDENT_REL_OVERRIDES.get(sid, {})
        _rel_look_down = _overrides.get("REL_LOOK_DOWN", self.cfg.REL_LOOK_DOWN)
        _rel_yaw       = _overrides.get("REL_YAW",       self.cfg.REL_YAW)
        _rel_mar       = _overrides.get("REL_MAR",       self.cfg.REL_MAR)

        if result.is_calibrated:
            result.looking_down  = dev_down >  _rel_look_down
            result.looking_left  = dev_yaw  < -_rel_yaw
            result.looking_right = dev_yaw  >  _rel_yaw
            result.talking       = (dev_mar > _rel_mar
                                    and mar_smoothed < 0.90
                                    and mar_variation > 0.008
                                    and mouth_open_vertical)
        else:
            result.looking_down  = False
            result.looking_left  = False
            result.looking_right = False
            result.talking       = (dev_mar > _rel_mar
                                    and mar_smoothed < 0.90
                                    and mar_variation > 0.008
                                    and mouth_open_vertical)


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3 — check_lean(): use per-student REL_LEAN override (line 349)
# ─────────────────────────────────────────────────────────────────────────────
# FIND inside check_lean():
#
#   if self.is_lean_calibrated(sid):
#       return abs(shoulder_diff_x - bl.lean) > self.cfg.REL_LEAN
#
# REPLACE:

        _overrides = self.cfg.STUDENT_REL_OVERRIDES.get(sid, {})
        _rel_lean  = _overrides.get("REL_LEAN", self.cfg.REL_LEAN)

        if self.is_lean_calibrated(sid):
            return abs(shoulder_diff_x - bl.lean) > _rel_lean
        else:
            return shoulder_diff_x < self.cfg.FALLBACK_LEAN


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 4 — Add method to AdaptiveLearner: set_student_thresholds() (end of class)
# ─────────────────────────────────────────────────────────────────────────────
# This is the clean public API that AutoThreshold calls after finalising a student.
# ADD this method to AdaptiveLearner:

    def set_student_thresholds(self, sid: str, overrides: dict):
        """
        Override the global REL_* values for a specific student.

        Parameters
        ----------
        sid       : student ID string
        overrides : dict with keys: REL_LOOK_DOWN, REL_YAW, REL_LEAN, REL_MAR

        Called by AutoThreshold.apply_to_learner() after calibration window closes.
        """
        self.cfg.STUDENT_REL_OVERRIDES[sid] = overrides
        print(
            f"[AdaptiveLearner] Per-student thresholds set for {sid}: "
            + ", ".join(f"{k}={v:.4f}" for k, v in overrides.items())
        )
