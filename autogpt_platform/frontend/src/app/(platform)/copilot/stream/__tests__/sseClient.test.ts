import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";

import { fetchSse, readSseFrames, type SseFrame } from "../sseClient";

function bodyOf(parts: (string | Uint8Array)[]) {
  const encoder = new TextEncoder();
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const part of parts) {
        controller.enqueue(
          typeof part === "string" ? encoder.encode(part) : part,
        );
      }
      controller.close();
    },
  });
}

async function framesOf(parts: (string | Uint8Array)[]) {
  const frames: SseFrame[] = [];
  await readSseFrames(bodyOf(parts), (frame) => frames.push(frame));
  return frames;
}

describe("readSseFrames", () => {
  it("tells entries, the route's own frames, comments and the terminator apart", async () => {
    const frames = await framesOf([
      'id: turn-1:5-0\ndata: {"type":"text-delta","id":"a","delta":"hi"}\n\n',
      ": heartbeat\n\n",
      'data: {"type":"error","errorText":"route failed"}\n\n',
      "data: [DONE]\n\n",
    ]);
    expect(frames).toEqual([
      {
        kind: "entry",
        entry: {
          turn: "turn-1",
          entryId: "5-0",
          chunk: { type: "text-delta", id: "a", delta: "hi" },
        },
      },
      { kind: "comment", text: "heartbeat" },
      {
        kind: "synthetic",
        chunk: { type: "error", errorText: "route failed" },
      },
      { kind: "done" },
    ]);
  });

  it("does not carry an entry's id over to the id-less frame after it", async () => {
    const frames = await framesOf([
      'id: t:1-0\ndata: {"type":"start","messageId":"m"}\n\n',
      'data: {"type":"finish"}\n\n',
    ]);
    expect(frames.map((f) => f.kind)).toEqual(["entry", "synthetic"]);
  });

  it("reassembles a frame, and a character, split across network chunks", async () => {
    const bytes = new TextEncoder().encode(
      'id: t:1-0\ndata: {"type":"text-delta","id":"a","delta":"…"}\n\n',
    );
    const ellipsis = bytes.indexOf(0xe2);
    const frames = await framesOf([
      bytes.slice(0, ellipsis + 1),
      bytes.slice(ellipsis + 1),
    ]);
    expect(frames).toEqual([
      expect.objectContaining({
        entry: expect.objectContaining({
          chunk: expect.objectContaining({ delta: "…" }),
        }),
      }),
    ]);
  });
});

describe("fetchSse", () => {
  it("returns a refused resume's status and body instead of frames", async () => {
    server.use(
      http.get("http://sse.test/stream", () =>
        HttpResponse.json(
          { reason: "trimmed", checkpoint: { entry_id: "9-0" } },
          { status: 409 },
        ),
      ),
    );
    const frames: SseFrame[] = [];
    const result = await fetchSse("http://sse.test/stream", {}, (f) =>
      frames.push(f),
    );
    expect(result).toEqual({
      ok: false,
      status: 409,
      body: { reason: "trimmed", checkpoint: { entry_id: "9-0" } },
    });
    expect(frames).toEqual([]);
  });
});
