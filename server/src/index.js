const express = require("express");
const cors    = require("cors");
const { v4: uuidv4 } = require("uuid");

const app  = express();
const PORT = 8080;

const store = {
  sessions: {},
  alerts:   [],
};

app.use(cors());
app.use(express.json());

function ts() { return new Date().toISOString(); }

app.get("/health", (req, res) => {
  res.json({ status: "ok", time: ts() });
});

app.post("/api/proctoring/alert", (req, res) => {
  const { examId, roomId, events } = req.body;
  if (!examId || !roomId || !Array.isArray(events)) {
    return res.status(400).json({ error: "Invalid payload" });
  }
  const sessionKey = `${examId}::${roomId}`;
  if (!store.sessions[sessionKey]) {
    store.sessions[sessionKey] = { examId, roomId, startedAt: ts(), events: [] };
  }
  const enriched = events.map((e) => ({
    id: uuidv4(), examId, roomId,
    type: e.type, studentId: e.trackId || null,
    confidence: e.confidence ?? null, distance: e.distance ?? null,
    timestamp: e.timestamp ? new Date(e.timestamp * 1000).toISOString() : ts(),
    receivedAt: ts(),
  }));
  store.sessions[sessionKey].events.push(...enriched);
  store.alerts.push(...enriched);
  enriched.forEach((e) => {
    console.log(`[${e.receivedAt}] 🚨 ${e.type.padEnd(25)} | ${e.examId} | student:${e.studentId}`);
  });
  return res.json({ received: enriched.length, sessionKey });
});

app.get("/api/proctoring/alerts", (req, res) => {
  const { examId, roomId, studentId, type, limit = "100" } = req.query;
  let results = [...store.alerts];
  if (examId)    results = results.filter((a) => a.examId    === examId);
  if (roomId)    results = results.filter((a) => a.roomId    === roomId);
  if (studentId) results = results.filter((a) => a.studentId === studentId);
  if (type)      results = results.filter((a) => a.type      === type);
  results.sort((a, b) => new Date(b.receivedAt) - new Date(a.receivedAt));
  results = results.slice(0, parseInt(limit, 10));
  res.json({ count: results.length, alerts: results });
});

app.get("/api/proctoring/summary", (req, res) => {
  const { examId, roomId } = req.query;
  if (!examId || !roomId) return res.status(400).json({ error: "examId and roomId required" });
  const session = store.sessions[`${examId}::${roomId}`];
  if (!session) return res.status(404).json({ error: "Session not found" });
  const studentMap = {};
  for (const evt of session.events) {
    const sid = evt.studentId || "UNKNOWN";
    if (!studentMap[sid]) studentMap[sid] = { studentId: sid, totalEvents: 0, byType: {} };
    studentMap[sid].totalEvents += 1;
    studentMap[sid].byType[evt.type] = (studentMap[sid].byType[evt.type] || 0) + 1;
  }
  res.json({ examId, roomId, sessionStart: session.startedAt, totalEvents: session.events.length, students: Object.values(studentMap) });
});

app.get("/api/proctoring/sessions", (_req, res) => {
  const list = Object.entries(store.sessions).map(([key, s]) => ({
    sessionKey: key, examId: s.examId, roomId: s.roomId,
    startedAt: s.startedAt, eventCount: s.events.length,
  }));
  res.json({ sessions: list });
});

app.delete("/api/proctoring/session", (req, res) => {
  const { examId, roomId } = req.query;
  const sessionKey = `${examId}::${roomId}`;
  if (!store.sessions[sessionKey]) return res.status(404).json({ error: "Session not found" });
  store.alerts = store.alerts.filter((a) => !(a.examId === examId && a.roomId === roomId));
  delete store.sessions[sessionKey];
  res.json({ deleted: sessionKey });
});

app.listen(PORT, () => {
  console.log(`\n🟢  Proctoring API server running on http://0.0.0.0:${PORT}`);
  console.log("   POST   /api/proctoring/alert");
  console.log("   GET    /api/proctoring/alerts");
  console.log("   GET    /api/proctoring/summary");
  console.log("   GET    /api/proctoring/sessions\n");
});
