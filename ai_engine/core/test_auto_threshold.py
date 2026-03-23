"""
test_auto_threshold.py
======================
Offline test — no camera or MediaPipe needed.
Run with:  python test_auto_threshold.py
"""

import time
import numpy as np
from auto_threshold import AutoThreshold, AutoThresholdConfig


def make_signals(n: int, gaze_mean=0.50, lean_mean=0.14, noise=0.01) -> list:
    """Generate n fake signal dicts simulating a calm, centred student."""
    rng = np.random.default_rng(42)
    return [
        {
            "gaze":      float(gaze_mean  + rng.normal(0, noise)),
            "look_down": float(0.030      + rng.normal(0, noise * 0.5)),
            "lean":      float(lean_mean  + rng.normal(0, noise)),
            "mar":       float(0.015      + rng.normal(0, noise * 0.3)),
            "head":      float(0.50       + rng.normal(0, noise)),
        }
        for _ in range(n)
    ]


def test_basic_calibration():
    print("\n── Test 1: basic 3-second calibration ──────────────────────────")
    cfg = AutoThresholdConfig(window_sec=0.2, fps=30, min_samples=10, debug=True)
    at  = AutoThreshold(cfg)
    sid = "S001"

    signals = make_signals(60)

    for s in signals:
        at.feed(sid, s)

    # Force time to pass by monkey-patching start time
    at._buffers[sid].window_start -= 1.0  # pretend 1 second elapsed

    assert at.should_finalise(sid), "should be ready to finalise"
    thr = at.finalise(sid)
    assert at.is_ready(sid)

    g = at.get(sid, "gaze")
    assert g is not None
    assert not g.is_fallback, "should be personalised, not fallback"
    assert 0.45 < g.baseline < 0.55, f"baseline should be ~0.5, got {g.baseline:.4f}"
    assert g.thr_low < g.baseline < g.thr_high

    print(f"  gaze baseline = {g.baseline:.4f}  [{g.thr_low:.4f} – {g.thr_high:.4f}]  ✓")
    print("Test 1 PASSED ✓")


def test_fallback_on_no_data():
    print("\n── Test 2: fallback when no data available ──────────────────────")
    cfg = AutoThresholdConfig(window_sec=0.1, max_window_sec=0.15, min_samples=5, debug=True)
    at  = AutoThreshold(cfg)
    sid = "S002"

    # Feed nothing (student was occluded)
    at._get_buf(sid)  # create empty buffer
    at._buffers[sid].window_start -= 1.0  # exceed max window

    thr = at.finalise(sid)
    assert at.is_ready(sid)
    assert thr.is_fallback, "should be fallback when no data"
    g = at.get(sid, "gaze")
    assert g.is_fallback
    print(f"  fallback gaze baseline = {g.baseline:.4f}  ✓")
    print("Test 2 PASSED ✓")


def test_apply_to_learner_mock():
    print("\n── Test 3: apply_to_learner() ───────────────────────────────────")
    from dataclasses import dataclass

    @dataclass
    class MockBaseline:
        yaw: float       = 0.500
        look_down: float = 0.042
        lean: float      = 0.150
        mar: float       = 0.018
        sample_count: int      = 0
        lean_sample_count: int = 0
        last_updated: float    = 0.0

    class MockConfig:
        CALIBRATION_FRAMES      = 6
        CALIBRATION_FRAMES_LEAN = 6

    import core.adaptive_learning as al_mod
    # Temporarily patch Config
    _orig = al_mod.Config
    al_mod.Config = MockConfig

    cfg = AutoThresholdConfig(window_sec=0.1, min_samples=5, debug=False)
    at  = AutoThreshold(cfg)
    sid = "S003"
    for s in make_signals(40, gaze_mean=0.48, lean_mean=0.16):
        at.feed(sid, s)
    at._buffers[sid].window_start -= 1.0
    at.finalise(sid)

    bl = MockBaseline()
    result = at.apply_to_learner(sid, bl)
    assert result is True
    assert abs(bl.yaw - 0.48) < 0.05, f"yaw should be ~0.48, got {bl.yaw:.4f}"
    assert bl.sample_count >= MockConfig.CALIBRATION_FRAMES

    al_mod.Config = _orig  # restore
    print(f"  applied: yaw={bl.yaw:.4f}  ld={bl.look_down:.4f}  lean={bl.lean:.4f}  mar={bl.mar:.4f}  ✓")
    print("Test 3 PASSED ✓")


def test_iqr_outlier_removal():
    print("\n── Test 4: IQR outlier removal ──────────────────────────────────")
    cfg = AutoThresholdConfig(window_sec=0.1, min_samples=10, debug=False)
    at  = AutoThreshold(cfg)
    sid = "S004"

    # Normal signals with a few gross outliers injected
    signals = make_signals(50, gaze_mean=0.50)
    # Inject outliers
    signals[5]["gaze"]  = 0.99
    signals[10]["gaze"] = 0.02
    signals[20]["gaze"] = 0.98

    for s in signals:
        at.feed(sid, s)
    at._buffers[sid].window_start -= 1.0
    thr = at.finalise(sid)

    g = at.get(sid, "gaze")
    assert 0.45 < g.baseline < 0.55, f"outliers should be removed, baseline={g.baseline:.4f}"
    print(f"  gaze baseline after outlier removal = {g.baseline:.4f}  ✓")
    print("Test 4 PASSED ✓")


def test_reset():
    print("\n── Test 5: reset() clears state ─────────────────────────────────")
    cfg = AutoThresholdConfig(window_sec=0.1, min_samples=5, debug=False)
    at  = AutoThreshold(cfg)
    sid = "S005"

    for s in make_signals(30):
        at.feed(sid, s)
    at._buffers[sid].window_start -= 1.0
    at.finalise(sid)
    assert at.is_ready(sid)

    at.reset(sid)
    assert not at.is_ready(sid)
    assert sid not in at._buffers
    print("Test 5 PASSED ✓")


if __name__ == "__main__":
    print("=" * 55)
    print("  AutoThreshold — Unit Tests")
    print("=" * 55)
    try:
        test_basic_calibration()
        test_fallback_on_no_data()
        test_apply_to_learner_mock()
        test_iqr_outlier_removal()
        test_reset()
        print("\n" + "=" * 55)
        print("  ALL TESTS PASSED ✓")
        print("=" * 55)
    except AssertionError as e:
        print(f"\n✗ ASSERTION FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        raise
