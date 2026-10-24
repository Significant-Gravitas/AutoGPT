// One copilot turn, consumed as the browser consumes it: POST the message to
// the session's stream route and read the Server-Sent Events until the turn
// finishes. The timings are what the user feels:
//
// - copilot_first_chunk_ms: the first frame of any kind ("Message received");
// - copilot_ttft_ms: the first text delta (or reasoning delta or tool call),
//   the moment something starts appearing in the chat;
// - copilot_turn_ms: the `finish` frame, when the reply is complete and saved.
//
// Each frame is `data: <json>` with a `type`; `data: [DONE]` ends the stream;
// `: heartbeat` and `: usage` comments arrive with empty data. A session that
// already has a turn running, or a user at the running-turn cap, gets a 200
// stream with no text (the message is queued), counted as queued rather than
// as a completed turn.
import sse from "k6/x/sse";
import { check } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";

import { getEnvironmentConfig } from "../configs/environment.js";
import { authHeaders } from "./auth.js";

const config = getEnvironmentConfig();

export const copilotFirstChunk = new Trend("copilot_first_chunk_ms", true);
export const copilotTtft = new Trend("copilot_ttft_ms", true);
export const copilotTurnTime = new Trend("copilot_turn_ms", true);
export const copilotChunks = new Trend("copilot_chunks_per_turn");
export const copilotTurns = new Counter("copilot_turns");
export const copilotTurnsQueued = new Counter("copilot_turns_queued");
export const copilotTurnsFailed = new Counter("copilot_turns_failed");
export const copilotTurnErrors = new Rate("copilot_turn_errors");

// The reply the dummy engine streams for any message, used to confirm the
// target runs it before the run sends turns at it.
export const DUMMY_REPLY = "I counted: 1... 2... 3. All done!";

const STREAM_ROUTE = "/api/chat/sessions/{session_id}/stream";

// Sends `message` to the session and reads the stream to its end. Returns
// the turn's outcome and timings; records the metrics when `record` is true.
export function streamTurn(sessionId, message, options) {
  const opts = options || {};
  const phase = opts.phase || "steady";
  const tags = { phase, endpoint: "copilot.stream" };
  const startedAt = Date.now();
  const turn = {
    status: 0,
    outcome: "timeout",
    firstChunkMs: null,
    ttftMs: null,
    turnMs: null,
    chunks: 0,
    textDeltas: 0,
    text: "",
    error: null,
    finish: false,
    done: false,
    followUpQueued: false,
  };

  const res = sse.open(
    `${config.API_BASE_URL}/chat/sessions/${sessionId}/stream`,
    {
      method: "POST",
      body: JSON.stringify({ message, is_user_message: true }),
      headers: opts.headers || authHeaders(opts.data),
      // A hard cap on the whole stream: without it a turn that never
      // finishes (no copilot executor running) holds the VU for an hour.
      timeout: opts.timeout || "120s",
      tags: { name: STREAM_ROUTE, endpoint: "copilot.stream", phase },
    },
    (client) => {
      client.on("event", (event) => {
        const raw = event.data;
        if (!raw) return;
        const now = Date.now();
        if (turn.firstChunkMs === null) turn.firstChunkMs = now - startedAt;
        if (raw === "[DONE]") {
          turn.done = true;
          client.close();
          return;
        }
        let frame;
        try {
          frame = JSON.parse(raw);
        } catch (e) {
          turn.error = turn.error || "unparseable frame";
          return;
        }
        turn.chunks++;
        switch (frame.type) {
          case "text-delta":
            if (turn.ttftMs === null) turn.ttftMs = now - startedAt;
            turn.textDeltas++;
            turn.text += frame.delta || "";
            break;
          case "reasoning-delta":
          case "tool-input-start":
            if (turn.ttftMs === null) turn.ttftMs = now - startedAt;
            break;
          case "data-follow-up-queued":
            turn.followUpQueued = true;
            break;
          case "error":
            turn.error = frame.errorText || "error frame";
            break;
          case "finish":
            turn.finish = true;
            turn.turnMs = now - startedAt;
            break;
          default:
            break;
        }
      });
      client.on("error", (e) => {
        const text = String(e.error());
        // The parser reports a line it does not know as an error; the stream
        // itself is fine, so keep reading.
        if (/unknown event/i.test(text)) return;
        turn.error = turn.error || text;
        client.close();
      });
    },
  );

  turn.status = res ? res.status : 0;
  if (turn.status !== 200) {
    turn.outcome = "rejected";
    turn.error = turn.error || (res && res.error) || `HTTP ${turn.status}`;
  } else if (turn.error) {
    turn.outcome = "failed";
  } else if (turn.finish && turn.textDeltas > 0) {
    turn.outcome = "completed";
  } else if (turn.finish) {
    turn.outcome = "queued";
  } else {
    turn.outcome = "timeout";
    turn.error = "stream ended without a finish frame";
  }

  if (opts.record !== false) recordTurn(turn, tags);
  return turn;
}

function recordTurn(turn, tags) {
  copilotTurns.add(1, tags);
  check(
    turn,
    {
      "copilot.stream: status 200": (t) => t.status === 200,
      "copilot.stream: turn completed or queued": (t) =>
        t.outcome === "completed" || t.outcome === "queued",
    },
    tags,
  );
  if (turn.outcome === "completed") {
    copilotFirstChunk.add(turn.firstChunkMs, tags);
    copilotTtft.add(turn.ttftMs, tags);
    copilotTurnTime.add(turn.turnMs, tags);
    copilotChunks.add(turn.chunks, tags);
    copilotTurnErrors.add(false, tags);
  } else if (turn.outcome === "queued") {
    copilotTurnsQueued.add(1, tags);
    copilotTurnErrors.add(false, tags);
  } else {
    copilotTurnsFailed.add(1, tags);
    copilotTurnErrors.add(true, tags);
    console.warn(
      `copilot turn ${turn.outcome} after ${turn.chunks} frames: ${turn.error}`,
    );
  }
}
