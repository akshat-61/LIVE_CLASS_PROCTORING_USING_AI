const express = require("express");
const cors    = require("cors");
const { v4: uuidv4 } = require("uuid");

const app  = express();
const PORT = 8080;

// ─── In-memory store (replace with a DB for production) ─────────────────────
const store = {
  sessions: {},   // sessionId → { examId, roomId, startedAt, events[] }
  alerts:   [],   // flat list for dashboard queries
};

// ─── Middleware ──────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json());

// ─── Logging helper ──────────────────────────────────────────────────────────
function ts() {
  return new Date().toISOString();
}

// ════════════════════════════════════════════════════════════════════════════
// ROUTES
// ════════════════════════════════════════════════════════════════════════════

// ── Health check ─────────────────────────────────────────────────────────────
app.get("/health", (req, res) => {
  res.json({ status: "ok", time: ts() });
});

// ── POST /api/proctoring/alert ───────────────────────────────────────────────
// Receives batched events from the Python AI engine.
// Body: { examId, roomId, events: [{ type, timestamp, confidence, trackId, distance }] }
app.post("/api/proctoring/alert", (req, res) => {
  const { examId, roomId, events } = req.body;

  if (!examId || !roomId || !Array.isArray(events)) {
    return res.status(400).json({ error: "Invalid payload" });
  }

  // Create a session key
  const sessionKey = `${examId}::${roomId}`;
  if (!store.sessions[sessionKey]) {
    store.sessions[sessionKey] = {
      examId,
      roomId,
      startedAt: ts(),
      events: [],
    };
  }

  const enriched = events.map((e) => ({
    id:         uuidv4(),
    examId,
    roomId,
    type:       e.type,
    studentId:  e.trackId  || null,
    confidence: e.confidence ?? null,
    distance:   e.distance  ?? null,
    timestamp:  e.timestamp
      ? new Date(e.timestamp * 1000).toISOString()
      : ts(),
    receivedAt: ts(),
  }));

  store.sessions[sessionKey].events.push(...enriched);
  store.alerts.push(...enriched);

  // Console summary
  enriched.forEach((e) => {
    console.log(`[${e.receivedAt}] 🚨 ${e.type.padEnd(25)} | ${e.examId} | ${e.roomId} | student:${e.studentId}`);
  });

  return res.json({ received: enriched.length, sessionKey });
});

// ── GET /api/proctoring/alerts ───────────────────────────────────────────────
// Query params: examId, roomId, studentId, type, limit (default 100)
app.get("/api/proctoring/alerts", (req, res) => {
  const { examId, roomId, studentId, type, limit = "100" } = req.query;
  let results = [...store.alerts];

  if (examId)    results = results.filter((a) => a.examId    === examId);
  if (roomId)    results = results.filter((a) => a.roomId    === roomId);
  if (studentId) results = results.filter((a) => a.studentId === studentId);
  if (type)      results = results.filter((a) => a.type      === type);

  // Most recent first
  results.sort((a, b) => new Date(b.receivedAt) - new Date(a.receivedAt));
  results = results.slice(0, parseInt(limit, 10));

  res.json({ count: results.length, alerts: results });
});

// ── GET /api/proctoring/summary ──────────────────────────────────────────────
// Returns per-student event counts + risk breakdown for an exam session.
app.get("/api/proctoring/summary", (req, res) => {
  const { examId, roomId } = req.query;

  if (!examId || !roomId) {
    return res.status(400).json({ error: "examId and roomId are required" });
  }

  const sessionKey = `${examId}::${roomId}`;
  const session    = store.sessions[sessionKey];

  if (!session) {
    return res.status(404).json({ error: "Session not found" });
  }

  // Aggregate per student
  const studentMap = {};
  for (const evt of session.events) {
    const sid = evt.studentId || "UNKNOWN";
    if (!studentMap[sid]) {
      studentMap[sid] = { studentId: sid, totalEvents: 0, byType: {} };
    }
    studentMap[sid].totalEvents += 1;
    studentMap[sid].byType[evt.type] = (studentMap[sid].byType[evt.type] || 0) + 1;
  }

  res.json({
    examId,
    roomId,
    sessionStart: session.startedAt,
    totalEvents:  session.events.length,
    students:     Object.values(studentMap),
  });
});

// ── GET /api/proctoring/sessions ─────────────────────────────────────────────
// Lists all active sessions.
app.get("/api/proctoring/sessions", (_req, res) => {
  const list = Object.entries(store.sessions).map(([key, s]) => ({
    sessionKey: key,
    examId:     s.examId,
    roomId:     s.roomId,
    startedAt:  s.startedAt,
    eventCount: s.events.length,
  }));
  res.json({ sessions: list });
});

// ── DELETE /api/proctoring/session ───────────────────────────────────────────
// Clear a session (e.g. after exam ends).
app.delete("/api/proctoring/session", (req, res) => {
  const { examId, roomId } = req.query;
  const sessionKey = `${examId}::${roomId}`;

  if (!store.sessions[sessionKey]) {
    return res.status(404).json({ error: "Session not found" });
  }

  // Also remove from flat alert list
  store.alerts = store.alerts.filter(
    (a) => !(a.examId === examId && a.roomId === roomId)
  );
  delete store.sessions[sessionKey];

  res.json({ deleted: sessionKey });
});

// ─── Start ───────────────────────────────────────────────────────────────────
app.listen(PORT, "0.0.0.0", () => {
  console.log(`\n🟢  Proctoring API server running on http://0.0.0.0:${PORT}`);
  console.log("   Endpoints:");
  console.log("   POST   /api/proctoring/alert      ← Python AI engine sends alerts here");
  console.log("   GET    /api/proctoring/alerts      ← Query all alerts");
  console.log("   GET    /api/proctoring/summary     ← Per-student summary for an exam");
  console.log("   GET    /api/proctoring/sessions    ← List active sessions");
  console.log("   DELETE /api/proctoring/session     ← End / clear a session");
  console.log("   GET    /health\n");
});