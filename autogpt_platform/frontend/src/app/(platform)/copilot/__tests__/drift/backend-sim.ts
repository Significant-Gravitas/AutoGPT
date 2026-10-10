import { getGetV2GetSessionMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { SessionDetailResponseMessagesItem } from "@/app/api/__generated__/models/sessionDetailResponseMessagesItem";
import { server } from "@/mocks/mock-server";
import { waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import fs from "node:fs";
import path from "node:path";
import { expect } from "vitest";
import { compareEntryIds } from "../../stream/turnConverter";
import {
  renderHost,
  TEST_BACKEND_BASE_URL,
  TEST_SESSION_ID,
  typeAndSend,
} from "../sse-helpers";

export type Row = SessionDetailResponseMessagesItem;

export interface RecordedTurn {
  frames: { id: string; sse: string }[];
  rows: Row[];
}

export const STREAM_URL = `${TEST_BACKEND_BASE_URL}/api/chat/sessions/${TEST_SESSION_ID}/stream`;
const FIXTURE_ROOT = path.resolve(
  process.cwd(),
  "../backend/test/fixtures/copilot_stream",
);
const SSE_HEADERS = {
  "content-type": "text/event-stream",
  "x-vercel-ai-ui-message-stream": "v1",
};
const HEARTBEAT_MS = 10_000;
const encoder = new TextEncoder();

/** A turn recorded from the backend pipeline by `stream_drift/recording.py`. */
export function loadRecordedTurn(name: string): RecordedTurn {
  const dir = path.join(FIXTURE_ROOT, name);
  const frames = fs
    .readFileSync(path.join(dir, "frames.jsonl"), "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  const rows = JSON.parse(fs.readFileSync(path.join(dir, "rows.json"), "utf8"));
  return { frames, rows };
}

interface TurnPlan {
  turn: RecordedTurn;
  /** Rows the session GET returns once this turn has ended; defaults to the
   *  recorded ones. A failed persist is a turn whose rows never landed. */
  persistedRows?: Row[];
  /** Start this turn the moment the previous one ends, before its finish is
   *  published, as a held call's continuation does. */
  chained?: boolean;
}

/**
 * The backend as the chat sees it: each turn's frames published one by one
 * under its own turn id, persisted at every checkpoint, served whole to a
 * resume without a cursor and from the cursor to one with it, and a session
 * GET whose rows, `active_stream` and checkpoint follow the turn.
 */
export function createBackendSim(plans: TurnPlan[]) {
  const state = {
    turnIndex: -1,
    trimmedBefore: 0,
    running: false,
    startedAt: "",
    committed: [] as Row[],
    current: [] as Row[],
  };
  const turns = plans.map(simulatedTurn);
  const published = new Map<number, number>();
  const connections: { closed: boolean; cut: () => void }[] = [];
  const wakers = new Set<() => void>();

  function startTurn(index: number) {
    state.turnIndex = index;
    state.startedAt = new Date().toISOString();
    state.trimmedBefore = 0;
    state.running = true;
    state.current = leadingUserRows(plans[index].turn.rows);
    published.set(index, 0);
  }

  // A checkpoint names the rows a persist landed; the GET reads them from then on.
  function persistThrough(index: number, end: number) {
    const rows = plans[index].persistedRows ?? plans[index].turn.rows;
    state.current = rows.filter((row) => Number(row.sequence ?? 0) < end);
  }

  // The backend persists, marks the turn completed and wakes any chained
  // turn before it publishes the finish.
  function endTurn(index: number) {
    const plan = plans[index];
    state.committed = [
      ...state.committed,
      ...(plan.persistedRows ?? plan.turn.rows),
    ];
    state.current = [];
    state.running = false;
    if (plans[index + 1]?.chained) startTurn(index + 1);
  }

  /** Publish the running turn's next `count` frames (default: all of them). */
  function publish(count = Infinity) {
    const index = state.turnIndex;
    const frames = turns[index].frames;
    let next = published.get(index) ?? 0;
    const target = Math.min(frames.length, next + count);
    while (next < target) {
      const checkpoint = turns[index].recordedCheckpoints.get(next);
      if (checkpoint)
        persistThrough(index, checkpoint.sequence + checkpoint.rows);
      if (isFinish(frames[next])) endTurn(index);
      next += 1;
      published.set(index, next);
    }
    wakers.forEach((wake) => wake());
  }

  function lastCheckpoint(index: number, before = Infinity) {
    const end = Math.min(published.get(index) ?? 0, before);
    for (let i = end - 1; i >= 0; i--) {
      const checkpoint = turns[index].checkpoints.get(i);
      if (checkpoint) return checkpoint;
    }
    return null;
  }

  function resume(turn: string, after: string, signal: AbortSignal) {
    const index = turns.findIndex((t) => t.id === turn);
    if (index === -1 || !published.has(index)) {
      connections.push({ closed: true, cut() {} });
      return HttpResponse.json({ reason: "expired" }, { status: 410 });
    }
    const frames = turns[index].frames;
    let from = frames.findIndex((f) => compareEntryIds(f.id, after) > 0);
    if (from === -1) from = frames.length;
    const floor = index === state.turnIndex ? state.trimmedBefore : 0;
    if (from < floor) {
      connections.push({ closed: true, cut() {} });
      return HttpResponse.json(
        { reason: "trimmed", checkpoint: lastCheckpoint(index, floor) },
        { status: 409 },
      );
    }
    return open(index, from, signal);
  }

  function open(
    turnIndex: number,
    from = state.trimmedBefore,
    signal?: AbortSignal,
  ) {
    const frames = turns[turnIndex].frames;
    let index = from;
    let cutRequested = false;
    let wake: (() => void) | null = null;
    const connection = {
      closed: false,
      cut() {
        cutRequested = true;
        wake?.();
      },
    };
    connections.push(connection);
    signal?.addEventListener("abort", connection.cut, { once: true });

    function close(controller: ReadableStreamDefaultController<Uint8Array>) {
      if (connection.closed) return;
      connection.closed = true;
      signal?.removeEventListener("abort", connection.cut);
      controller.close();
    }

    const stream = new ReadableStream<Uint8Array>({
      async pull(controller) {
        while (!cutRequested) {
          if (index < (published.get(turnIndex) ?? 0)) {
            const frame = frames[index++];
            controller.enqueue(encoder.encode(frame.sse));
            if (isFinish(frame)) {
              controller.enqueue(encoder.encode("data: [DONE]\n\n"));
              close(controller);
            }
            return;
          }
          const heartbeat = await new Promise<boolean>((resolve) => {
            const timer = setTimeout(() => resolve(true), HEARTBEAT_MS);
            wake = () => {
              clearTimeout(timer);
              resolve(false);
            };
            wakers.add(wake);
          });
          if (wake) wakers.delete(wake);
          wake = null;
          if (heartbeat) {
            controller.enqueue(encoder.encode(": heartbeat\n\n"));
            return;
          }
        }
        close(controller);
      },
      cancel() {
        connection.cut();
        connection.closed = true;
        signal?.removeEventListener("abort", connection.cut);
      },
    });
    return new HttpResponse(stream, { status: 200, headers: SSE_HEADERS });
  }

  return {
    handlers: [
      http.post(STREAM_URL, ({ request }) => {
        startTurn(state.turnIndex + 1);
        return open(state.turnIndex, undefined, request.signal);
      }),
      http.get(STREAM_URL, ({ request }) => {
        const params = new URL(request.url).searchParams;
        const turn = params.get("turn");
        if (turn !== null)
          return resume(turn, params.get("after") ?? "0-0", request.signal);
        if (!state.running) return new HttpResponse(null, { status: 204 });
        return open(state.turnIndex, undefined, request.signal);
      }),
    ],
    connections,
    /** Mount mid-turn: the turn is already running when the page loads. */
    beginRunning(index = 0) {
      startTurn(index);
    },
    publish,
    /** The running turn's stream lost its first `count` entries. */
    trimHead(count: number) {
      state.trimmedBefore = count;
    },
    cutOpenConnections() {
      connections.filter((c) => !c.closed).forEach((c) => c.cut());
    },
    session(): SessionDetailResponse {
      return {
        id: TEST_SESSION_ID,
        created_at: "2026-09-30T00:00:00Z",
        updated_at: "2026-09-30T00:00:00Z",
        user_id: "test-user",
        has_more_messages: false,
        oldest_sequence: null,
        metadata: { dry_run: false, builder_graph_id: null },
        expert_id: null,
        messages: numbered([...state.committed, ...state.current]),
        active_stream: state.running
          ? {
              turn_id: turns[state.turnIndex].id,
              last_message_id: "0-0",
              started_at: state.startedAt,
              checkpoint: lastCheckpoint(state.turnIndex),
            }
          : null,
        chat_status: state.running ? "running" : "idle",
      };
    },
    /** Every row a reload shows once the whole plan has run. */
    finalRows() {
      return numbered(
        plans.flatMap((plan) => plan.persistedRows ?? plan.turn.rows),
      );
    },
  };
}

export type BackendSim = ReturnType<typeof createBackendSim>;

/** Mount the chat against the simulator; the session GET follows its state. */
export function renderAgainst(sim: BackendSim) {
  server.use(...sim.handlers);
  return renderHost({
    sessionResponse: getGetV2GetSessionMockHandler200(() => sim.session()),
  });
}

/** Send the turn's own prompt and wait for its POST to reach the backend. */
export async function sendPrompt(sim: BackendSim, turn: RecordedTurn) {
  const before = sim.connections.length;
  await typeAndSend(String(turn.rows[0].content));
  await waitForConnections(sim, before + 1);
}

export async function waitForConnections(sim: BackendSim, count: number) {
  await waitFor(() => expect(sim.connections.length).toBe(count), {
    timeout: 10_000,
  });
}

/** Past the post-finish probe (500 ms) and the first reconnect delay (1 s),
 *  so the client decides while the backend turn is still running. */
export function turnKeepsRunning() {
  return new Promise((resolve) => setTimeout(resolve, 2500));
}

/** The full text of each text block the turn streamed, in order. */
export function streamedTextBlocks(turn: RecordedTurn) {
  const blocks = new Map<string, string>();
  for (const frame of turn.frames) {
    const data = frameData(frame);
    if (data === null) continue;
    const chunk = JSON.parse(data);
    if (chunk.type === "text-start") blocks.set(chunk.id, "");
    if (chunk.type === "text-delta") {
      blocks.set(chunk.id, (blocks.get(chunk.id) ?? "") + chunk.delta);
    }
  }
  return [...blocks.values()];
}

/**
 * A recorded turn as the sim serves it: under its own turn id, so a chained
 * turn's entries never sit at or before the cursor the previous one left,
 * and with its checkpoints' sequences moved to where its rows land here.
 */
function simulatedTurn(plan: TurnPlan, index: number, plans: TurnPlan[]) {
  const id = `drift-turn-${index}`;
  const base = plans
    .slice(0, index)
    .reduce((n, p) => n + (p.persistedRows ?? p.turn.rows).length, 0);
  const shift = base - Number(plan.turn.rows[0]?.sequence ?? 0);
  const recordedCheckpoints = new Map<number, CheckpointData>();
  const checkpoints = new Map<number, CheckpointData & { entry_id: string }>();
  const frames = plan.turn.frames.map((frame, i) => {
    let sse = frame.sse.replace(/^id: [^\n]*:/, `id: ${id}:`);
    const data = frameData(frame);
    const chunk = data
      ? (JSON.parse(data) as { type: string; data?: CheckpointData })
      : null;
    if (chunk?.type === "data-checkpoint" && chunk.data) {
      recordedCheckpoints.set(i, chunk.data);
      const moved = { ...chunk.data, sequence: chunk.data.sequence + shift };
      checkpoints.set(i, {
        entry_id: frame.id,
        rows: moved.rows,
        sequence: moved.sequence,
      });
      sse = `id: ${id}:${frame.id}\ndata: ${JSON.stringify({ ...chunk, data: moved })}\n\n`;
    }
    return { id: frame.id, sse };
  });
  return { id, frames, checkpoints, recordedCheckpoints };
}

interface CheckpointData {
  rows: number;
  sequence: number;
  digest?: string;
}

function isFinish(frame: { sse: string }) {
  return frameData(frame)?.startsWith('{"type":"finish"}') ?? false;
}

/** The JSON on a frame's `data:` line; entry frames lead with an `id:` line. */
function frameData(frame: { sse: string }) {
  const line = frame.sse.split("\n").find((l) => l.startsWith("data: "));
  return line ? line.slice("data: ".length) : null;
}

function leadingUserRows(rows: Row[]) {
  const firstOther = rows.findIndex((row) => row.role !== "user");
  return firstOther === -1 ? rows : rows.slice(0, firstOther);
}

function numbered(rows: Row[]): Row[] {
  return rows.map((row, sequence) => ({
    ...row,
    id: `row-${sequence}`,
    sequence,
  }));
}
