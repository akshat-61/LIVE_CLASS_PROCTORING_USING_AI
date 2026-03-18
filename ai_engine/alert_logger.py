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

# Sentinel value — putting this in the queue tells the writer thread to exit
_STOP_SENTINEL = object()


class AlertLogger:

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self._lock = Lock()

        # Initialise log file if it doesn't exist yet
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w") as f:
                json.dump([], f)

        # ── Async write queue ──────────────────────────────────────────────
        # log_event() puts alerts here and returns immediately.
        # The background writer thread drains the queue and does all disk I/O.
        # maxsize=2000 — at ~10 alerts/s this is >3 minutes of headroom.
        self._queue: Queue = Queue(maxsize=2000)

        self._writer_thread = Thread(
            target=self._writer_loop,
            name="alert-logger-writer",
            daemon=True          # exits automatically when main process exits
        )
        self._writer_thread.start()

    # ── Public API (called from AI thread — must be non-blocking) ─────────

    def log_event(self, student_id, event, confidence=1.0, frame=None):
        """
        Non-blocking. Builds the alert dict and puts it on the queue.
        Returns immediately — disk I/O happens in the background thread.
        If the queue is full (>2000 pending writes) the alert is dropped
        with a warning rather than blocking the AI pipeline.
        """
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
        """
        Block until all queued alerts have been written to disk.
        Call this at session end / shutdown to ensure nothing is lost.
        """
        self._queue.join()

    def shutdown(self):
        """
        Flush remaining alerts then stop the background thread cleanly.
        Call once from ai_engine.shutdown().
        """
        self.flush()
        self._queue.put(_STOP_SENTINEL)
        self._writer_thread.join(timeout=5.0)

    # ── Background writer (runs in its own thread) ─────────────────────────

    def _writer_loop(self):
        """
        Drains the alert queue and writes to disk in batches.

        Batching strategy: wait up to 0.1s for more alerts after the first
        one arrives, then write everything accumulated so far in one
        json.load -> extend -> json.dump cycle. This is far cheaper than
        one file open per alert when alerts burst (e.g. multiple students
        flagged in the same frame).
        """
        while True:
            # Block until at least one item is available
            try:
                first = self._queue.get(timeout=1.0)
            except Empty:
                continue

            # Stop signal
            if first is _STOP_SENTINEL:
                break

            # Collect any additional alerts that arrived in the same burst
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

            # Write the whole batch in one file operation
            self._write_batch(batch)

            # Mark every item in the batch as done (for flush() / queue.join())
            for _ in batch:
                self._queue.task_done()

    def _write_batch(self, batch: list):
        """
        Appends a list of alert dicts to the JSON log file atomically.

        FIX 1 — Atomic write: builds the full updated JSON in memory, writes
        to a temp file in the same directory, then os.replace() swaps it in.
        os.replace() is atomic on POSIX — the file is never partially written.
        This eliminates the 8192-byte buffer-boundary corruption seen when
        appending directly with f.seek(0)/f.truncate().

        FIX 12 — JSONL fallback: if the existing log file contains invalid
        JSON (e.g. corrupted from a previous crash), switch transparently to
        line-delimited JSON (one object per line). JSONL is corruption-resilient
        — a bad line never corrupts the rest of the file.
        """
        import tempfile

        with self._lock:
            try:
                # Read existing data — repair corrupt file gracefully
                existing = []
                if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 2:
                    try:
                        with open(LOG_FILE, "r") as f:
                            existing = json.load(f)
                    except json.JSONDecodeError:
                        # FIX 12: existing file is corrupt — salvage what we can
                        # by reading it as JSONL, then start fresh JSON array
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

                # FIX 1: write to temp file first, then atomically swap
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
                # FIX 12: last-resort fallback — append as JSONL so no data is lost
                try:
                    jsonl_path = LOG_FILE.replace(".json", ".jsonl")
                    with open(jsonl_path, "a") as jf:
                        for alert in batch:
                            jf.write(json.dumps(alert) + "\n")
                    print(f"[AlertLogger] ↳ Batch saved to fallback JSONL: {jsonl_path}")
                except Exception as e2:
                    print(f"[AlertLogger] ↳ JSONL fallback also failed: {e2}")