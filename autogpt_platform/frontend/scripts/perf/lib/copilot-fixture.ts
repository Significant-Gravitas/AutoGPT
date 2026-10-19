import fs from "fs";
import path from "path";
import { FIXTURES_DIR } from "./paths";
import { fulfillJSON, type RouteHandlers } from "./routes";
import { readFixture, splitFrames, type StreamServer } from "./stream-server";

const FIXED_TIME = "2026-01-05T09:00:00.000Z";
const STREAM_TIMEOUT_MS = 120_000;
const TURN_START_TIMEOUT_MS = 15_000;
const POLL_MS = 50;

export interface SessionSpec {
  id: string;
  title: string;
  messageCount: number;
}

interface SessionRow {
  role: string;
  content: string;
  sequence: number;
  [key: string]: unknown;
}

interface Args {
  userID: string;
  sessions: SessionSpec[];
  stream: StreamServer;
  replyFixture: string;
  paceMs: number;
}

export const SESSION_L = "00000000-0000-4000-8000-0000000000a1";
export const SESSION_M = "00000000-0000-4000-8000-0000000000a2";
export const SESSION_S = "00000000-0000-4000-8000-0000000000a3";

export const COPILOT_SESSIONS: SessionSpec[] = [
  { id: SESSION_L, title: "Perf: long thread (200)", messageCount: 200 },
  { id: SESSION_M, title: "Perf: medium thread (50)", messageCount: 50 },
  { id: SESSION_S, title: "Perf: short thread (10)", messageCount: 10 },
];

function loadSessionRows(): SessionRow[] {
  const file = path.join(FIXTURES_DIR, "session-200.json");
  return (JSON.parse(fs.readFileSync(file, "utf8")) as { messages: [] })
    .messages;
}

/** The reply's text, from the `data:` line of each frame (recorded frames
 *  also carry `id:` lines and `: comment` heartbeats). */
export function replyText(sse: string) {
  return splitFrames(sse)
    .flatMap((frame) => frame.split("\n"))
    .filter((line) => line.startsWith("data: {"))
    .map((line) => line.slice("data: ".length))
    .map((data) => JSON.parse(data) as { type: string; delta?: string })
    .filter((event) => event.type === "text-delta")
    .map((event) => event.delta ?? "")
    .join("");
}

/**
 * Serves the copilot session list, session detail and stream from fixtures.
 * Stateful on purpose: once a streamed turn ends, the session detail returns
 * the new user and assistant rows, as the backend would, so the page's
 * post-turn refetch does not wipe the reply it just streamed.
 */
export function createCopilotFixture({
  userID,
  sessions,
  stream,
  replyFixture,
  paceMs,
}: Args) {
  const allRows = loadSessionRows();
  const reply = replyText(readFixture(replyFixture));
  const extraRows = new Map<string, SessionRow[]>();
  const pendingTurns: string[] = [];
  let turnCount = 0;

  function find(id: string) {
    return sessions.find((session) => session.id === id);
  }

  function rowsFor(spec: SessionSpec) {
    const rows = [
      ...allRows.slice(0, spec.messageCount),
      ...(extraRows.get(spec.id) ?? []),
    ];
    return rows.map((row) => ({ ...row, session_id: spec.id }));
  }

  function detail(spec: SessionSpec) {
    const messages = rowsFor(spec);
    return {
      id: spec.id,
      created_at: FIXED_TIME,
      updated_at: FIXED_TIME,
      user_id: userID,
      chat_status: "idle",
      messages,
      active_stream: null,
      has_more_messages: false,
      oldest_sequence: messages[0]?.sequence ?? null,
      total_prompt_tokens: 0,
      total_completion_tokens: 0,
      metadata: { dry_run: false, kind: "normal" },
      expert_id: null,
    };
  }

  function summary(spec: SessionSpec) {
    return {
      id: spec.id,
      created_at: FIXED_TIME,
      updated_at: FIXED_TIME,
      title: spec.title,
      chat_status: "idle",
      is_processing: false,
      source_platform: null,
      is_pinned: false,
      expert_id: null,
    };
  }

  function appendTurn(spec: SessionSpec, message: string) {
    const rows = rowsFor(spec);
    const next = (rows.at(-1)?.sequence ?? -1) + 1;
    const added = [
      { role: "user", content: message, sequence: next },
      { role: "assistant", content: reply, sequence: next + 1 },
    ].map((row) => ({
      id: `perf-turn-${spec.id}-${row.sequence}`,
      tool_calls: null,
      tool_call_id: null,
      duration_ms: null,
      created_at: FIXED_TIME,
      metadata: {},
      ...row,
    }));
    extraRows.set(spec.id, [...(extraRows.get(spec.id) ?? []), ...added]);
  }

  const handlers: RouteHandlers = {
    chatSessionList: async (route) => {
      if (route.request().method() !== "GET") return route.fallback();
      await fulfillJSON(route, {
        sessions: sessions.map(summary),
        total: sessions.length,
      });
    },
    chatSession: async (route, match) => {
      const spec = find(match[1]);
      if (!spec || route.request().method() !== "GET") return route.fallback();
      const olderPage = new URL(route.request().url()).searchParams.has(
        "before_sequence",
      );
      await fulfillJSON(
        route,
        olderPage
          ? { ...detail(spec), messages: [], has_more_messages: false }
          : detail(spec),
      );
    },
    chatSessionChild: async (route, match) => {
      if (!find(match[1])) return route.fallback();
      const child = match[2];
      if (child === "messages/pending") {
        return fulfillJSON(route, { messages: [], count: 0 });
      }
      if (child === "at-reset") return fulfillJSON(route, { jobs: [] });
      if (child === "share/state") {
        return fulfillJSON(route, { is_shared: false, share_token: null });
      }
      await fulfillJSON(route, {});
    },
    chatStream: async (route, match) => {
      const spec = find(match[1]);
      const request = route.request();
      if (!spec) return route.fallback();
      if (request.method() !== "POST") {
        return route.fulfill({ status: 204 });
      }
      const tag = `${spec.id}:${++turnCount}`;
      const body = request.postDataJSON() as { message?: string } | null;
      pendingTurns.push(tag);
      stream
        .waitForDone(tag, STREAM_TIMEOUT_MS)
        .then(() => appendTurn(spec, body?.message ?? ""))
        .catch(() => undefined);
      await route.continue({
        url: stream.url({ fixture: replyFixture, paceMs, tag }),
      });
    },
  };

  return {
    handlers,
    /** Resolves once turn number `turn` (1-based) has been fully streamed. */
    async waitForTurn(turn: number) {
      const deadline = Date.now() + TURN_START_TIMEOUT_MS;
      while (pendingTurns.length < turn) {
        if (Date.now() > deadline) {
          throw new Error(`Copilot turn ${turn} never reached the stream`);
        }
        await new Promise((resolve) => setTimeout(resolve, POLL_MS));
      }
      await stream.waitForDone(pendingTurns[turn - 1], STREAM_TIMEOUT_MS);
    },
  };
}
