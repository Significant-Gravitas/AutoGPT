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
  /** `id` is the stored entry's; a comment frame carries none on the wire. */
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
 * The backend as the chat sees it: a turn's frames published one by one into
 * its stream, and a session GET whose rows and `active_stream` follow the
 * turn. A GET naming a turn serves that turn's entries after its `after`
 * cursor, or refuses as the route does: 409 when entries after the cursor
 * were trimmed, 410 for a turn it never ran. A GET without one replays the
 * running turn from its first held entry, as older clients read it.
 */
export function createBackendSim(plans: TurnPlan[]) {
  const state = {
    turnIndex: -1,
    trimmedBefore: 0,
    running: false,
    startedAt: "",
    rows: [] as Row[],
  };
  const published = new Map<number, number>();
  const connections: { closed: boolean; cut: () => void }[] = [];
  const wakers = new Set<() => void>();
  const resumes: Record<string, string>[] = [];

  function startTurn(index: number) {
    state.turnIndex = index;
    state.startedAt = new Date().toISOString();
    state.trimmedBefore = 0;
    state.running = true;
    state.rows = [...state.rows, ...leadingUserRows(plans[index].turn.rows)];
    published.set(index, 0);
  }

  // The backend persists, marks the turn completed and wakes any chained
  // turn before it publishes the finish.
  function endTurn(index: number) {
    const plan = plans[index];
    const alreadyPersisted = leadingUserRows(plan.turn.rows).length;
    const turnRows = plan.persistedRows ?? plan.turn.rows;
    state.rows = [...state.rows, ...turnRows.slice(alreadyPersisted)];
    state.running = false;
    if (plans[index + 1]?.chained) startTurn(index + 1);
  }

  /** Publish the running turn's next `count` frames (default: all of them). */
  function publish(count = Infinity) {
    const index = state.turnIndex;
    const frames = plans[index].turn.frames;
    let next = published.get(index) ?? 0;
    const target = Math.min(frames.length, next + count);
    while (next < target) {
      if (isFinish(frames[next])) endTurn(index);
      next += 1;
      published.set(index, next);
    }
    wakers.forEach((wake) => wake());
  }

  function open(turnIndex: number, from: number) {
    const frames = plans[turnIndex].turn.frames;
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

    function close(controller: ReadableStreamDefaultController<Uint8Array>) {
      connection.closed = true;
      controller.close();
    }

    const stream = new ReadableStream<Uint8Array>({
      async pull(controller) {
        while (!cutRequested) {
          if (index < (published.get(turnIndex) ?? 0)) {
            const frame = frames[index++];
            controller.enqueue(encoder.encode(onTurn(frame.sse, turnIndex)));
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
      },
    });
    return new HttpResponse(stream, { status: 200, headers: SSE_HEADERS });
  }

  return {
    handlers: [
      http.post(STREAM_URL, () => {
        startTurn(state.turnIndex + 1);
        return open(state.turnIndex, 0);
      }),
      http.get(STREAM_URL, ({ request }) => {
        const params = new URL(request.url).searchParams;
        resumes.push(Object.fromEntries(params));
        const turn = params.get("turn");
        if (turn === null) {
          if (!state.running) return new HttpResponse(null, { status: 204 });
          return open(state.turnIndex, state.trimmedBefore);
        }
        const index = turnIndexOf(turn);
        if (index === null || !published.has(index)) {
          return HttpResponse.json({ reason: "expired" }, { status: 410 });
        }
        const frames = plans[index].turn.frames;
        const after = params.get("after") ?? "0-0";
        const from = frames.findIndex((f) => compareEntryIds(f.id, after) > 0);
        const start = from === -1 ? frames.length : from;
        if (index === state.turnIndex && start < state.trimmedBefore) {
          return HttpResponse.json(
            {
              reason: "trimmed",
              checkpoint: checkpointAt(frames[state.trimmedBefore]),
            },
            { status: 409 },
          );
        }
        return open(index, start);
      }),
    ],
    /** The query of every GET, in order. */
    resumes,
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
        messages: numbered(state.rows),
        active_stream: state.running
          ? {
              turn_id: turnIdOf(state.turnIndex),
              last_message_id: "0-0",
              started_at: state.startedAt,
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

function isFinish(frame: { sse: string }) {
  return frameData(frame)?.startsWith('{"type":"finish"}') ?? false;
}

/** The JSON on a frame's `data:` line; entry frames lead with an `id:` line. */
function frameData(frame: { sse: string }) {
  const line = frame.sse.split("\n").find((l) => l.startsWith("data: "));
  return line ? line.slice("data: ".length) : null;
}

function turnIdOf(index: number) {
  return `drift-turn-${index}`;
}

function turnIndexOf(turnId: string) {
  const match = /^drift-turn-(\d+)$/.exec(turnId);
  return match ? Number(match[1]) : null;
}

/** Recorded frames name one turn; each turn the plan runs gets its own id. */
function onTurn(sse: string, index: number) {
  return sse.replace(/^id: [^\n]*:/, `id: ${turnIdOf(index)}:`);
}

/** The refusal's checkpoint: the first held entry, when it is one. */
function checkpointAt(frame: { id: string; sse: string } | undefined) {
  const data = frame ? frameData(frame) : null;
  const chunk = data ? JSON.parse(data) : null;
  if (chunk?.type !== "data-checkpoint") return null;
  return {
    entry_id: frame!.id,
    rows: chunk.data.rows,
    sequence: chunk.data.sequence,
  };
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
