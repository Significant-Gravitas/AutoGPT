import { describe, expect, it, vi } from "vitest";

import { TurnStream } from "../turnStream";
import { loadRecordedTurn } from "./recordedTurns";

vi.mock("@sentry/nextjs", () => ({ captureMessage: vi.fn() }));

function frame(entry: number, data: unknown) {
  return `id: t:${entry}-0\ndata: ${JSON.stringify(data)}\n\n`;
}

const TEXT_TURN = [
  frame(1, { type: "start", messageId: "m1" }),
  frame(2, { type: "start-step" }),
  frame(3, { type: "text-start", id: "b" }),
  frame(4, { type: "text-delta", id: "b", delta: "Hello " }),
  frame(5, { type: "text-delta", id: "b", delta: "world" }),
  frame(6, { type: "text-end", id: "b" }),
  frame(7, { type: "finish-step" }),
  frame(8, { type: "finish" }),
];

function sse(frames: string[]) {
  return new Response(frames.join(""), {
    headers: { "content-type": "text/event-stream" },
  });
}

function makeStream({
  first,
  resumes,
}: {
  first: () => Response;
  resumes: ((query: string) => Response)[];
}) {
  const queries: string[] = [];
  const stream = new TurnStream({
    sessionId: "s1",
    turnId: null,
    isSend: true,
    dropStartMessageId: false,
    openFirst: async () => first(),
    openResume: async (query) => {
      queries.push(query);
      const next = resumes.shift();
      if (!next) throw new TypeError("network");
      return next(query);
    },
  });
  return { stream, queries };
}

async function readAll(readable: ReadableStream<unknown>) {
  const chunks: { type: string; delta?: string }[] = [];
  const reader = readable.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) return chunks;
    chunks.push(value as { type: string; delta?: string });
  }
}

async function phase(stream: TurnStream, expected: string) {
  await vi.waitFor(() => expect(stream.getState().phase).toBe(expected));
}

describe("TurnStream", () => {
  it("reconnects from its cursor and hands the parser every entry once", async () => {
    const { stream, queries } = makeStream({
      first: () => sse(TEXT_TURN.slice(0, 4)),
      // The server replays from the start: the cursor drops the overlap.
      resumes: [() => sse(TEXT_TURN)],
    });
    const readable = await stream.open();
    const chunks = readAll(readable!);
    await phase(stream, "lost");
    stream.reconnect();

    const types = (await chunks).map((c) => c.type);
    expect(types).toEqual([
      "start",
      "start-step",
      "text-start",
      "text-delta",
      "text-delta",
      "text-end",
      "finish-step",
      "finish",
    ]);
    expect(queries).toEqual(["?turn=t&after=4-0"]);
    expect(stream.getState().phase).toBe("finished");
  });

  it("continues after the checkpoint a 409 names, and asks for a hydrate", async () => {
    const { stream, queries } = makeStream({
      first: () => sse(TEXT_TURN.slice(0, 2)),
      resumes: [
        () =>
          Response.json(
            { reason: "trimmed", checkpoint: { entry_id: "7-0" } },
            { status: 409 },
          ),
        () => sse(TEXT_TURN.slice(7)),
      ],
    });
    const readable = await stream.open();
    const chunks = readAll(readable!);
    await phase(stream, "lost");
    stream.reconnect();

    expect((await chunks).map((c) => c.type)).toEqual([
      "start",
      "start-step",
      "finish",
    ]);
    expect(queries).toEqual(["?turn=t&after=2-0", "?turn=t&after=7-0"]);
    expect(stream.getState()).toMatchObject({
      phase: "finished",
      verified: false,
    });
  });

  it("ends the turn on 410 without a finish of its own", async () => {
    const { stream } = makeStream({
      first: () => sse(TEXT_TURN.slice(0, 2)),
      resumes: [() => Response.json({ reason: "expired" }, { status: 410 })],
    });
    const readable = await stream.open();
    const chunks = readAll(readable!);
    await phase(stream, "lost");
    stream.reconnect();

    expect((await chunks).map((c) => c.type)).toEqual(["start", "start-step"]);
    expect(stream.getState()).toMatchObject({
      phase: "finished",
      verified: false,
    });
  });

  it("keeps a delta whose block it never saw start away from the parser", async () => {
    const { stream } = makeStream({
      first: () => sse([TEXT_TURN[0], ...TEXT_TURN.slice(4)]),
      resumes: [],
    });
    const chunks = await readAll((await stream.open())!);
    expect(chunks.map((c) => c.type)).toEqual([
      "start",
      "finish-step",
      "finish",
    ]);
  });

  it("verifies a finished turn against its checkpoint digest", async () => {
    const turn = loadRecordedTurn("dummy-text-turn");
    const { stream } = makeStream({
      first: () => sse(turn.sse),
      resumes: [],
    });
    await readAll((await stream.open())!);
    expect(stream.getState()).toMatchObject({
      phase: "finished",
      verified: true,
    });
  });

  it("does not verify a turn whose checkpoint digest disagrees", async () => {
    const turn = loadRecordedTurn("dummy-text-turn");
    const tampered = turn.sse.map((f) =>
      f.replace(/"digest": "[0-9a-f]+"/, '"digest": "0000"'),
    );
    const { stream } = makeStream({ first: () => sse(tampered), resumes: [] });
    await readAll((await stream.open())!);
    expect(stream.getState().verified).toBe(false);
  });

  it("rejects a send the backend refused, with its body as the message", async () => {
    const { stream } = makeStream({
      first: () => new Response('{"detail":"usage limit"}', { status: 429 }),
      resumes: [],
    });
    await expect(stream.open()).rejects.toThrow('{"detail":"usage limit"}');
  });

  it("closes from this side without reconnecting", async () => {
    const { stream, queries } = makeStream({
      first: () => sse(TEXT_TURN.slice(0, 3)),
      resumes: [],
    });
    const chunks = readAll((await stream.open())!);
    stream.close();
    await chunks;
    expect(stream.getState().phase).toBe("closed");
    stream.reconnect();
    expect(queries).toEqual([]);
  });
});
