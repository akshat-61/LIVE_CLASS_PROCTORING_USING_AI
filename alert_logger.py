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
        Appends a list of alert dicts to the JSON log file in one operation.
        """
        with self._lock:
            try:
                with open(LOG_FILE, "r+") as f:
                    data = json.load(f)
                    data.extend(batch)
                    f.seek(0)
                    f.truncate()
                    json.dump(data, f, indent=4)
            except Exception as e:
                print(f"[AlertLogger] ERROR writing batch: {e}")