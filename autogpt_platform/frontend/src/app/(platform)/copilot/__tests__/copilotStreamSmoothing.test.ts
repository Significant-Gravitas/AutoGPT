import type { UIMessageChunk } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MAX_LAG_MS, TICK_MS } from "../copilotStreamPacer";
import {
  createSmoothingTransform,
  type SmoothingOptions,
} from "../copilotStreamSmoothing";

let hidden = false;

beforeEach(() => {
  hidden = false;
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

// A live stream: the test pushes chunks and moves the clock; `out` fills as
// the transform emits.
function liveStream(options: SmoothingOptions = {}) {
  let source!: ReadableStreamDefaultController<UIMessageChunk>;
  const out: UIMessageChunk[] = [];
  const reader = new ReadableStream<UIMessageChunk>({
    start(controller) {
      source = controller;
    },
  })
    .pipeThrough(
      createSmoothingTransform({
        now: () => Date.now(),
        isHidden: () => hidden,
        tickMs: TICK_MS,
        ...options,
      }),
    )
    .getReader();
  const done = (async () => {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) return;
      out.push(value);
    }
  })();
  return {
    out,
    reader,
    push: (chunk: UIMessageChunk) => source.enqueue(chunk),
    async end() {
      source.close();
      await vi.runAllTimersAsync();
      await done;
    },
  };
}

function textDelta(delta: string, id = "t1"): UIMessageChunk {
  return { type: "text-delta", id, delta };
}

function joinedText(chunks: UIMessageChunk[]) {
  return chunks
    .filter((c) => c.type === "text-delta")
    .map((c) => (c.type === "text-delta" ? c.delta : ""))
    .join("");
}

function textDeltas(chunks: UIMessageChunk[]) {
  return chunks.flatMap((c) => (c.type === "text-delta" ? [c.delta] : []));
}

const WORDS_300 = `${Array.from({ length: 300 }, (_, i) => `w${i}`).join(" ")} `;

describe("createSmoothingTransform", () => {
  it("splits a bursty delta into word-sized deltas without losing text", async () => {
    const stream = liveStream();
    const burst = "The quick brown fox jumps over the lazy dog ";
    stream.push(textDelta(burst));
    await vi.advanceTimersByTimeAsync(MAX_LAG_MS);

    expect(textDeltas(stream.out).length).toBeGreaterThan(1);
    expect(joinedText(stream.out)).toBe(burst);
    await stream.end();
  });

  it("holds back a trailing partial word until more text arrives", async () => {
    const stream = liveStream();
    stream.push(textDelta("Hello wor"));
    await vi.advanceTimersByTimeAsync(500);
    expect(joinedText(stream.out)).toBe("Hello ");

    stream.push(textDelta("ld! Bye "));
    await vi.advanceTimersByTimeAsync(500);
    expect(joinedText(stream.out)).toBe("Hello world! Bye ");
    for (const delta of textDeltas(stream.out)) expect(delta).toMatch(/\s$/);
    await stream.end();
  });

  it("flushes the remaining partial word when the stream closes", async () => {
    const stream = liveStream();
    stream.push(textDelta("no trailing space"));
    await stream.end();
    expect(joinedText(stream.out)).toBe("no trailing space");
  });

  it("paces a large burst instead of dumping half of it in a few frames", async () => {
    const stream = liveStream();
    stream.push(textDelta(WORDS_300));

    await vi.advanceTimersByTimeAsync(160);
    expect(joinedText(stream.out).length).toBeLessThan(WORDS_300.length * 0.3);

    await vi.advanceTimersByTimeAsync(MAX_LAG_MS / 2 - 160);
    const halfway = joinedText(stream.out).length / WORDS_300.length;
    expect(halfway).toBeGreaterThan(0.35);
    expect(halfway).toBeLessThan(0.65);

    await vi.advanceTimersByTimeAsync(MAX_LAG_MS / 2 + TICK_MS);
    expect(joinedText(stream.out)).toBe(WORDS_300);
    expect(textDeltas(stream.out).length).toBeGreaterThan(30);
    await stream.end();
  });

  it("never falls more than the lag cap behind a fast model", async () => {
    const stream = liveStream();
    // 2,000 characters a second, five times the steady rate.
    let sent = "";
    for (let t = 0; t < 3_000; t += 20) {
      const chunk = "abcdefghi ".repeat(4);
      sent += chunk;
      stream.push(textDelta(chunk));
      await vi.advanceTimersByTimeAsync(20);
      const arrivedLagAgo = Math.max(0, ((t - MAX_LAG_MS) / 20) * 40);
      expect(joinedText(stream.out).length).toBeGreaterThanOrEqual(
        arrivedLagAgo - 80,
      );
    }
    await stream.end();
    expect(joinedText(stream.out)).toBe(sent);
  });

  it("catches up in one step when the timer fires late", async () => {
    // A background tab clamps timers to about a second.
    const stream = liveStream({
      wait: () => new Promise((resolve) => setTimeout(resolve, 1_000)),
    });
    stream.push(textDelta(WORDS_300));
    await vi.advanceTimersByTimeAsync(1_000);
    expect(joinedText(stream.out)).toBe(WORDS_300);
    await stream.end();
  });

  it("shows everything complete at once while the tab is hidden", async () => {
    hidden = true;
    const stream = liveStream();
    stream.push(textDelta(`${WORDS_300}tail`));
    await vi.advanceTimersByTimeAsync(0);
    expect(joinedText(stream.out)).toBe(WORDS_300);
    await stream.end();
  });

  it("flushes at once when the stream finishes", async () => {
    const stream = liveStream();
    stream.push(textDelta(WORDS_300));
    stream.push({ type: "finish" });
    await vi.advanceTimersByTimeAsync(0);
    expect(joinedText(stream.out)).toBe(WORDS_300);
    expect(stream.out[stream.out.length - 1].type).toBe("finish");
    await stream.end();
  });

  it("paces a whole block sent in one delta before its text-end", async () => {
    const stream = liveStream();
    stream.push({ type: "text-start", id: "t1" });
    stream.push(textDelta(WORDS_300));
    stream.push({ type: "text-end", id: "t1" });
    stream.push({ type: "tool-input-start", toolCallId: "c1", toolName: "x" });

    await vi.advanceTimersByTimeAsync(160);
    expect(stream.out.some((c) => c.type === "text-end")).toBe(false);

    await vi.advanceTimersByTimeAsync(MAX_LAG_MS);
    const types = stream.out.map((c) => c.type);
    expect(joinedText(stream.out)).toBe(WORDS_300);
    expect(types[0]).toBe("text-start");
    expect(types.slice(-2)).toEqual(["text-end", "tool-input-start"]);
    await stream.end();
  });

  it("emits buffered text before a non-delta chunk passes through", async () => {
    const stream = liveStream();
    stream.push({ type: "text-start", id: "t1" });
    stream.push(textDelta("almost done"));
    stream.push({ type: "text-end", id: "t1" });
    await stream.end();

    const types = stream.out.map((c) => c.type);
    expect(types[0]).toBe("text-start");
    expect(types[types.length - 1]).toBe("text-end");
    expect(joinedText(stream.out)).toBe("almost done");
  });

  it("smooths reasoning deltas and drains them before the part switch", async () => {
    const stream = liveStream();
    stream.push({ type: "reasoning-delta", id: "r1", delta: "thinking hard " });
    stream.push(textDelta("answer here "));
    await stream.end();

    const types = stream.out.map((c) => c.type);
    expect(
      stream.out
        .map((c) => (c.type === "reasoning-delta" ? c.delta : ""))
        .join(""),
    ).toBe("thinking hard ");
    expect(joinedText(stream.out)).toBe("answer here ");
    expect(types.lastIndexOf("reasoning-delta")).toBeLessThan(
      types.indexOf("text-delta"),
    );
  });

  it("passes deltas with providerMetadata through unsplit", async () => {
    const stream = liveStream();
    const chunk: UIMessageChunk = {
      type: "text-delta",
      id: "t1",
      delta: "one two three ",
      providerMetadata: { test: { keep: true } },
    };
    stream.push(chunk);
    await stream.end();
    expect(stream.out).toEqual([chunk]);
  });

  it("keeps markdown whitespace intact across newlines", async () => {
    const stream = liveStream();
    const burst = "# Title\n\nParagraph one.\n\n- item a\n- item b\n";
    stream.push(textDelta(burst));
    await stream.end();
    expect(joinedText(stream.out)).toBe(burst);
  });

  it.each([
    ["Japanese", "これは日本語の文章です。𠮷野家でがくせいと話しました。"],
    ["Chinese", "我们今天去公园散步吧。天气很好，阳光明媚，大家都很开心。"],
  ])("reveals %s text written without spaces word by word", async (_, text) => {
    const stream = liveStream();
    stream.push(textDelta(text));
    await vi.advanceTimersByTimeAsync(MAX_LAG_MS + TICK_MS);

    // Split on whitespace, the whole text would wait for the stream's end.
    const deltas = textDeltas(stream.out);
    expect(deltas.length).toBeGreaterThanOrEqual(4);
    expect(joinedText(stream.out).length).toBeGreaterThan(text.length * 0.8);
    for (const delta of deltas) {
      expect(delta).not.toMatch(/^[\uDC00-\uDFFF]|[\uD800-\uDBFF]$|^\p{M}/u);
    }
    await stream.end();
    expect(joinedText(stream.out)).toBe(text);
  });

  it("tears down cleanly when the reader cancels mid-drain", async () => {
    const stream = liveStream();
    stream.push(textDelta(WORDS_300));
    await vi.advanceTimersByTimeAsync(100);
    const shown = stream.out.length;
    expect(shown).toBeGreaterThan(0);

    await stream.reader.cancel();
    await vi.advanceTimersByTimeAsync(MAX_LAG_MS * 2);
    expect(stream.out.length).toBe(shown);
  });

  it("takes larger steps when reduced motion is asked for", async () => {
    vi.stubGlobal("matchMedia", (query: string) => ({
      matches: query.includes("reduce"),
    }));
    const reduced = liveStream({ tickMs: undefined });
    reduced.push(textDelta(WORDS_300));
    await vi.advanceTimersByTimeAsync(MAX_LAG_MS + 200);
    vi.unstubAllGlobals();
    const normal = liveStream();
    normal.push(textDelta(WORDS_300));
    await vi.advanceTimersByTimeAsync(MAX_LAG_MS + 200);

    expect(joinedText(reduced.out)).toBe(WORDS_300);
    expect(textDeltas(reduced.out).length).toBeLessThan(
      textDeltas(normal.out).length / 3,
    );
    await reduced.end();
    await normal.end();
  });
});

describe("createSmoothingTransform on a resumed stream", () => {
  it("shows the replayed turn at once and paces what arrives live after it", async () => {
    const stream = liveStream({ replayUntilQuietMs: 200 });
    const replay: UIMessageChunk[] = [
      { type: "start" },
      { type: "text-start", id: "t1" },
      textDelta(WORDS_300),
    ];
    for (const chunk of replay) stream.push(chunk);
    await vi.advanceTimersByTimeAsync(0);
    expect(stream.out).toEqual(replay);

    await vi.advanceTimersByTimeAsync(500);
    stream.push(textDelta(WORDS_300));
    await vi.advanceTimersByTimeAsync(160);
    const live = joinedText(stream.out).length - WORDS_300.length;
    expect(live).toBeGreaterThan(0);
    expect(live).toBeLessThan(WORDS_300.length * 0.3);

    await vi.advanceTimersByTimeAsync(MAX_LAG_MS);
    expect(joinedText(stream.out)).toBe(WORDS_300 + WORDS_300);
    await stream.end();
  });
});
