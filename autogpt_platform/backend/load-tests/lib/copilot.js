// One copilot turn, consumed as the browser consumes it: POST the message to
// the session's stream route and read the Server-Sent Events until the turn
// finishes. The timings are what the user feels:
//
// - copilot_first_chunk_ms: the first frame of any kind ("Message received");
// - copilot_ttft_ms: the first text delta (or reasoning delta or tool call),
//   the moment something starts appearing in the chat;
// - copilot_turn_ms: the `finish` frame, when the reply is complete and saved.
//
// Every turn's metrics carry an `engine` tag, the engine the account's turns
// run on (dummy, scripted or real: lib/auth.js), so each engine is judged on
// its own numbers.
//
// Each frame is `data: <json>` with a `type`; `data: [DONE]` ends the stream;
// `: heartbeat` and `: usage` comments arrive with empty data. A session that
// already has a turn running gets a 200 stream with no text and a
// `data-follow-up-queued` frame: the message is queued, and the turn counts
// as queued rather than completed. Any other stream that finishes without
// text counts as failed. A user at the running-turn cap gets one too, with
// nothing to say so, and the journeys keep every account well under that
// cap, so an empty reply is never mistaken for a queued one.
import sse from "k6/x/sse";
import { check } from "k6";
import { Counter, Rate, Trend } from "k6/metrics";

import { getEnvironmentConfig } from "../configs/environment.js";
import { authHeaders } from "./auth.js";
import { randomUuid } from "./random.js";

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
// target runs it for the dummy accounts before the run sends turns at it.
export const DUMMY_REPLY = "I counted: 1... 2... 3. All done!";

// The script the scripted model plays for the scripted accounts' turns,
// model-scripts/loadtest-turn.json, the same file a preview's scripted model
// reads. A message picks it with a [[script:loadtest-turn#<nonce>]] marker.
// The script answers a conversation by how many replies it already holds, so
// every scripted turn starts a new session.
export const SCRIPT_NAME = "loadtest-turn";
const SCRIPT = JSON.parse(open("../model-scripts/loadtest-turn.json"));
// What the script's last turn says, which ends every scripted turn.
export const SCRIPTED_FINAL_TEXT = SCRIPT.turns[SCRIPT.turns.length - 1].text;
// What the script's command prints in the sandbox.
export const SCRIPTED_TOOL_OUTPUT = "load-test";

// The message the frontend sends as a hire's first turn, with expert_kickoff
// set on the stream request (frontend/src/app/(platform)/copilot/
// expertKickoff.ts).
export const KICKOFF_PROMPT =
  "You were just hired. Call expert_onboarding once, and nothing else, this " +
  "turn: a one-sentence greeting introducing yourself in your voice, with " +
  "no dashes, then " +
  "2-3 short questions the user can answer by tapping. Every question and " +
  "every option must come from your own role and area of expertise and from " +
  "the workflows installed on you — ask which outcome in your area to start " +
  "with, and which of the services your work depends on you should be " +
  "connected to, naming the real providers rather than asking in the " +
  "abstract. Do not ask about work outside your role, whatever else you " +
  "know about the user or their team. Do not run a workflow, create a " +
  "schedule, or start any other work before the answers come back.";

const STREAM_ROUTE = "/api/chat/sessions/{session_id}/stream";
// How much of each tool's output a turn keeps, for a probe to read.
const TOOL_OUTPUT_CHARS = 2000;

// A scripted turn's message: the marker, with a nonce that finds the turn in
// the scripted model's journal, then the text.
export function scriptedMessage(nonce, text) {
  return `[[script:${SCRIPT_NAME}#${nonce}]] ${text}`;
}

// Whether a turn's reply holds the script's final text. The platform relays
// the text of both model turns, the first one's before the tool call and the
// last one's after it.
export function hasScriptedReply(turn) {
  return squash(turn.text).indexOf(squash(SCRIPTED_FINAL_TEXT)) >= 0;
}

function squash(text) {
  return String(text || "")
    .replace(/\s+/g, " ")
    .trim();
}

// What one real-model turn costs, for the estimate a run that sends them
// prints and reports: COPILOT_REAL_COST_PER_TURN_USD, by default $0.07,
// production's median for a one-step turn.
export function realTurnCostUsd() {
  const raw = __ENV.COPILOT_REAL_COST_PER_TURN_USD;
  if (raw === undefined || raw === "") return 0.07;
  const cost = Number(raw);
  if (!(cost >= 0)) {
    throw new Error(
      `COPILOT_REAL_COST_PER_TURN_USD must be a number of dollars, got "${raw}"`,
    );
  }
  return cost;
}

// Sends `message` to the session and reads the stream to its end. Returns
// the turn's outcome and timings; records the metrics, tagged with
// `options.engine` (the engine the account's turns run on), unless
// `options.record` is false. The turn is `options.account`'s (one of
// data.accounts), the VU's own account's, or sent with `options.headers`.
export function streamTurn(sessionId, message, options) {
  const opts = options || {};
  const phase = opts.phase || "steady";
  const engine = opts.engine;
  if (!engine) throw new Error("streamTurn needs the turn's engine");
  // A hire's kickoff waits on the expert's setup, so it is judged apart.
  const kind = opts.expertKickoff === true ? "kickoff" : "chat";
  const tags = { phase, endpoint: "copilot.stream", engine, turn: kind };
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
    toolCalls: 0,
    toolOutputs: [],
    error: null,
    finish: false,
    done: false,
    followUpQueued: false,
  };

  const res = sse.open(
    `${config.API_BASE_URL}/chat/sessions/${sessionId}/stream`,
    {
      method: "POST",
      // As the browser sends it: a fresh idempotency key per send, and the
      // kickoff flag on a hire's first turn.
      body: JSON.stringify({
        message,
        is_user_message: true,
        message_id: randomUuid(),
        expert_kickoff: opts.expertKickoff === true,
      }),
      headers:
        opts.headers ||
        authHeaders(opts.data, opts.account ? opts.account.index : undefined),
      // A hard cap on the whole stream: without it a turn that never
      // finishes (no copilot executor running) holds the VU for an hour.
      timeout: opts.timeout || "120s",
      tags: { name: STREAM_ROUTE, endpoint: "copilot.stream", phase, engine },
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
          case "tool-input-start":
            turn.toolCalls++;
            if (turn.ttftMs === null) turn.ttftMs = now - startedAt;
            break;
          case "reasoning-delta":
            if (turn.ttftMs === null) turn.ttftMs = now - startedAt;
            break;
          case "tool-output-available":
            turn.toolOutputs.push(
              (typeof frame.output === "string"
                ? frame.output
                : JSON.stringify(frame.output) || ""
              ).slice(0, TOOL_OUTPUT_CHARS),
            );
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
  } else if (turn.finish && turn.followUpQueued) {
    turn.outcome = "queued";
  } else if (turn.finish) {
    turn.outcome = "failed";
    turn.error = "the turn finished without a reply";
  } else {
    turn.outcome = "timeout";
    turn.error = "stream ended without a finish frame";
  }

  if (opts.record !== false) recordTurn(turn, tags);
  return turn;
}

// A turn that could not start, its session not created: counted as failed,
// so a journey that starts none fails copilot_turn_errors instead of passing
// with no samples.
export function recordUnstartedTurn(engine, phase) {
  const tags = { phase, endpoint: "copilot.stream", engine, turn: "chat" };
  copilotTurns.add(1, tags);
  copilotTurnsFailed.add(1, tags);
  copilotTurnErrors.add(true, tags);
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
      `copilot ${tags.engine} turn ${turn.outcome} after ${turn.chunks} frames: ${turn.error}`,
    );
  }
}
