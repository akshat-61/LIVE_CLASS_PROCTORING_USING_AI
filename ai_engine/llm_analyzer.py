import os
import json
import logging
import threading
from collections import defaultdict

log = logging.getLogger(__name__)

LLM_PROVIDER   = os.environ.get("LLM_PROVIDER", "openai")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_URL     = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL   = os.environ.get("OLLAMA_MODEL", "llama3")

_analysis_cache: dict = {}
_lock = threading.Lock()


def _build_student_prompt(sid: str, events: list, score: float, seat_pos=None) -> str:
    event_counts = defaultdict(int)
    for e in events:
        etype = e.get("type") or e.get("event_type") or str(e)
        event_counts[etype] += 1

    sorted_events = sorted(event_counts.items(), key=lambda x: -x[1])
    event_lines = "\n".join(f"  - {etype}: {cnt} times" for etype, cnt in sorted_events)

    seat_info = f"Seat position: {seat_pos}" if seat_pos else "Seat position: unknown"
    risk_level = (
        "CRITICAL" if score >= 80 else
        "HIGH"     if score >= 40 else
        "MEDIUM"   if score >= 15 else
        "LOW"
    )

    return f"""You are an AI exam proctoring analyst. Analyse the following student data and write a concise 2-3 sentence professional summary for an exam report.

Student ID: {sid}
Risk Score: {score:.1f} / 100  ({risk_level})
{seat_info}
Total events detected: {len(events)}

Event breakdown:
{event_lines}

Write a factual, professional summary. Do not use the student's name. Focus on what the behavioral data suggests. Be specific about which behaviors occurred most frequently. Do not speculate beyond the data."""


def _build_room_prompt(all_students: list, total_events: int, exam_duration_min: int) -> str:
    high_risk = [s for s in all_students if s["score"] >= 40]
    critical  = [s for s in all_students if s["score"] >= 80]

    return f"""You are an AI exam proctoring analyst. Write a concise exam session summary for the invigilator's report.

Exam duration: {exam_duration_min} minutes
Total students monitored: {len(all_students)}
Total behavioral events detected: {total_events}
High-risk students (score >= 40): {len(high_risk)}
Critical-risk students (score >= 80): {len(critical)}

High-risk student IDs and scores:
{chr(10).join(f"  {s['sid']}: {s['score']:.1f}" for s in sorted(high_risk, key=lambda x: -x['score'])[:10])}

Write a 3-4 sentence executive summary of the exam session. Highlight overall patterns, identify if cheating appeared isolated or coordinated, and recommend next steps for the invigilator. Be professional and factual."""


def _call_openai(prompt: str) -> str:
    try:
        import openai
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        log.error("OpenAI call failed: %s", e)
        return f"[LLM analysis unavailable: {e}]"


def _call_ollama(prompt: str) -> str:
    try:
        import requests
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        log.error("Ollama call failed: %s", e)
        return f"[LLM analysis unavailable: {e}]"


def _call_llm(prompt: str) -> str:
    if LLM_PROVIDER == "ollama":
        return _call_ollama(prompt)
    if OPENAI_API_KEY:
        return _call_openai(prompt)
    return "[LLM analysis skipped — no API key configured. Set OPENAI_API_KEY or LLM_PROVIDER=ollama]"


def analyse_student(sid: str, events: list, score: float, seat_pos=None) -> str:
    cache_key = f"{sid}_{int(score)}_{len(events)}"
    with _lock:
        if cache_key in _analysis_cache:
            return _analysis_cache[cache_key]

    if len(events) < 3 or score < 5:
        result = "No significant behavioral events detected during this session."
    else:
        prompt = _build_student_prompt(sid, events, score, seat_pos)
        result = _call_llm(prompt)

    with _lock:
        _analysis_cache[cache_key] = result

    log.info("[LLM] Student %s analysis complete (%d chars)", sid, len(result))
    return result


def analyse_room(all_students: list, total_events: int, exam_duration_min: int = 60) -> str:
    prompt = _build_room_prompt(all_students, total_events, exam_duration_min)
    result = _call_llm(prompt)
    log.info("[LLM] Room analysis complete (%d chars)", len(result))
    return result


def analyse_students_async(students_data: list, callback):
    def worker():
        results = {}
        for s in students_data:
            sid    = s["sid"]
            events = s.get("events", [])
            score  = s.get("score", 0.0)
            pos    = s.get("seat_pos")
            results[sid] = analyse_student(sid, events, score, pos)
        callback(results)
    threading.Thread(target=worker, daemon=True).start()


def reset():
    with _lock:
        _analysis_cache.clear()
