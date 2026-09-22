import type { UIMessageChunk } from "ai";
import { describe, expect, it } from "vitest";

import { createSmoothingTransform } from "../copilotStreamSmoothing";

const instantWait = () => Promise.resolve();

async function pipe(chunks: UIMessageChunk[]): Promise<UIMessageChunk[]> {
  const source = new ReadableStream<UIMessageChunk>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(chunk);
      controller.close();
    },
  });

  const out: UIMessageChunk[] = [];
  const reader = source
    .pipeThrough(createSmoothingTransform(instantWait))
    .getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    out.push(value);
  }
  return out;
}

function textDelta(delta: string, id = "t1"): UIMessageChunk {
  return { type: "text-delta", id, delta };
}

function joinedText(chunks: UIMessageChunk[]) {
  return chunks
    .filter((c) => c.type === "text-delta")
    .map((c) => c.delta)
    .join("");
}

describe("createSmoothingTransform", () => {
  it("splits a bursty delta into word-sized deltas without losing text", async () => {
    const burst = "The quick brown fox jumps over the lazy dog ";
    const out = await pipe([textDelta(burst)]);

    const deltas = out.filter((c) => c.type === "text-delta");
    expect(deltas.length).toBeGreaterThan(1);
    expect(joinedText(out)).toBe(burst);
    for (const chunk of deltas) {
      expect(chunk.id).toBe("t1");
    }
  });

  it("holds back a trailing partial word until more text arrives", async () => {
    const out = await pipe([textDelta("Hello wor"), textDelta("ld! Bye ")]);
    const deltas = out.filter((c) => c.type === "text-delta");

    expect(joinedText(out)).toBe("Hello world! Bye ");
    // No emitted delta may end mid-word ("Hello wor" must never render):
    // every chunk ends on a whitespace boundary.
    for (const chunk of deltas) {
      expect(chunk.delta).toMatch(/\s$/);
    }
    expect(deltas.some((c) => c.delta.includes("world!"))).toBe(true);
  });

  it("flushes the remaining partial word when the stream closes", async () => {
    const out = await pipe([textDelta("no trailing space")]);
    expect(joinedText(out)).toBe("no trailing space");
  });

  it("flushes buffered text before passing a non-delta chunk through", async () => {
    const out = await pipe([
      { type: "text-start", id: "t1" },
      textDelta("almost done"),
      { type: "text-end", id: "t1" },
    ]);

    const types = out.map((c) => c.type);
    expect(types[0]).toBe("text-start");
    expect(types[types.length - 1]).toBe("text-end");
    expect(joinedText(out)).toBe("almost done");
    const endIndex = types.indexOf("text-end");
    const lastDeltaIndex = types.lastIndexOf("text-delta");
    expect(lastDeltaIndex).toBeLessThan(endIndex);
  });

  it("smooths reasoning deltas and flushes on part switch", async () => {
    const out = await pipe([
      { type: "reasoning-delta", id: "r1", delta: "thinking hard " },
      textDelta("answer here "),
    ]);

    const reasoning = out.filter((c) => c.type === "reasoning-delta");
    const text = out.filter((c) => c.type === "text-delta");
    expect(reasoning.map((c) => c.delta).join("")).toBe("thinking hard ");
    expect(text.map((c) => c.delta).join("")).toBe("answer here ");
    const lastReasoningIndex = out
      .map((c) => c.type)
      .lastIndexOf("reasoning-delta");
    const firstTextIndex = out.findIndex((c) => c.type === "text-delta");
    expect(lastReasoningIndex).toBeLessThan(firstTextIndex);
  });

  it("passes deltas with providerMetadata through unsplit", async () => {
    const chunk: UIMessageChunk = {
      type: "text-delta",
      id: "t1",
      delta: "one two three ",
      providerMetadata: { test: { keep: true } },
    };
    const out = await pipe([chunk]);
    expect(out).toEqual([chunk]);
  });

  it("keeps markdown whitespace intact across newlines", async () => {
    const burst = "# Title\n\nParagraph one.\n\n- item a\n- item b\n";
    const out = await pipe([textDelta(burst)]);
    expect(joinedText(out)).toBe(burst);
  });

  it("tears down cleanly when the reader cancels mid-drain", async () => {
    let release = () => {};
    const gatedWait = () =>
      new Promise<void>((resolve) => {
        release = resolve;
      });
    const source = new ReadableStream<UIMessageChunk>({
      start(controller) {
        controller.enqueue(textDelta("one two three four five "));
      },
    });

    const reader = source
      .pipeThrough(createSmoothingTransform(gatedWait))
      .getReader();
    const first = await reader.read();
    expect(first.done).toBe(false);

    // Cancel while the drain loop is parked on the gated wait, then let it
    // resume — it must exit without enqueueing into the cancelled stream.
    await reader.cancel();
    release();
    await new Promise((resolve) => setTimeout(resolve, 0));
  });

  it("drains large backlogs in bounded ticks via adaptive catch-up", async () => {
    const words = Array.from({ length: 300 }, (_, i) => `w${i}`).join(" ");
    const out = await pipe([textDelta(`${words} `)]);
    const deltas = out.filter((c) => c.type === "text-delta");

    expect(joinedText(out)).toBe(`${words} `);
    // 300 words at 1 word/tick would be 300 chunks; adaptive pacing emits
    // ~1/BACKLOG_DRAIN_TICKS of the backlog per tick (~76 ticks for 300).
    expect(deltas.length).toBeLessThanOrEqual(90);
    // The first tick after a burst carries multiple words (catch-up mode).
    expect(deltas[0].type === "text-delta" && deltas[0].delta).toMatch(
      /^(\S+\s+){2,}/,
    );
  });
});
