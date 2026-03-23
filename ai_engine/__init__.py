from ai.ai_engine import (
    run_ai_on_frame,
    reset_session,
    shutdown,
    generate_final_report,
    get_all_events,
    get_all_states,
    get_all_evidence,
    send_event_async,
    save_evidence,
    init_seat_zones,
    can_send,
    dist,
    AI_FRAME_SIZE,
    SEAT_ZONE_RADIUS,
    STABILITY_FRAMES,
    CUSTOM_CONF_THRESHOLD,
    CUSTOM_CLASS,
    PHONE_CONF_THRESHOLD,
    GAZE_LEFT_THRESHOLD,
    GAZE_RIGHT_THRESHOLD,
)

from ai.ai_engine import (
    score_engine,
    learner,
    person_model,
    pose_detector,
    student_id_map,
    seat_positions,
    locked,
    stable_counter,
    _lock_frame,
    _calib_accumulated_positions,
    EXAM_ID,
    ROOM_ID,
    ALERT_URL,
)

def __getattr__(name):
    import ai.ai_engine as _core
    return getattr(_core, name)

def __setattr__(name, value):
    import ai.ai_engine as _core
    setattr(_core, name, value)
