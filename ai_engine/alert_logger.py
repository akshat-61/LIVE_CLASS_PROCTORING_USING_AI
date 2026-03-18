import json
import os
from datetime import datetime
from threading import Lock, Thread
from queue import Queue, Empty

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

LOG_FILE = os.path.join(
    LOG_DIR,
    f"alerts_{datetime.now().strftime('%Y-%m-%d')}.json"
)

_STOP_SENTINEL = object()


class AlertLogger:

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._lock = Lock()

        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w") as f:
                json.dump([], f)

        self._queue: Queue = Queue(maxsize=2000)

        self._writer_thread = Thread(
            target=self._writer_loop,
            name="alert-logger-writer",
            daemon=True          # exits automatically when main process exits
        )
        self._writer_thread.start()

    def log_event(self, student_id, event, confidence=1.0, frame=None):
        alert = {
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "student_id": student_id,
            "event":      event,
            "confidence": confidence,
            "frame":      frame,
        }

        try:
            self._queue.put_nowait(alert)
        except Exception:
            # Queue full — drop rather than block
            print(f"[AlertLogger] ⚠️  Queue full — dropped event: {event} ({student_id})")

    def flush(self, timeout: float = 5.0):
        self._queue.join()

    def shutdown(self):
        self.flush()
        self._queue.put(_STOP_SENTINEL)
        self._writer_thread.join(timeout=5.0)

    def _writer_loop(self):
        while True:
            try:
                first = self._queue.get(timeout=1.0)
            except Empty:
                continue

            if first is _STOP_SENTINEL:
                break

            batch = [first]
            try:
                while True:
                    item = self._queue.get_nowait()
                    if item is _STOP_SENTINEL:
                        self._queue.put(_STOP_SENTINEL)
                        break
                    batch.append(item)
            except Empty:
                pass

            self._write_batch(batch)

            for _ in batch:
                self._queue.task_done()

    def _write_batch(self, batch: list):
        import tempfile

        with self._lock:
            try:
                existing = []
                if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 2:
                    try:
                        with open(LOG_FILE, "r") as f:
                            existing = json.load(f)
                    except json.JSONDecodeError:
                        print("[AlertLogger] ⚠ Log file corrupt — salvaging via JSONL repair")
                        salvaged = []
                        with open(LOG_FILE, "r") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    salvaged.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass  # skip truly unrecoverable lines
                        existing = salvaged
                        print(f"[AlertLogger] ✅ Salvaged {len(salvaged)} records")

                updated = existing + batch

                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=LOG_DIR, prefix=".alert_tmp_", suffix=".json"
                )
                try:
                    with os.fdopen(tmp_fd, "w") as tmp_f:
                        json.dump(updated, tmp_f, indent=4)
                    os.replace(tmp_path, LOG_FILE)   # atomic on POSIX
                except Exception:
                    # Clean up temp file if swap failed
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise

            except Exception as e:
                print(f"[AlertLogger] ERROR writing batch: {e}")
                try:
                    jsonl_path = LOG_FILE.replace(".json", ".jsonl")
                    with open(jsonl_path, "a") as jf:
                        for alert in batch:
                            jf.write(json.dumps(alert) + "\n")
                    print(f"[AlertLogger] ↳ Batch saved to fallback JSONL: {jsonl_path}")
                except Exception as e2:
                    print(f"[AlertLogger] ↳ JSONL fallback also failed: {e2}")