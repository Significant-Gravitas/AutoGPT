import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { MAX_LAG_MS, STEADY_CHARS_PER_SECOND } from "../../copilotStreamPacer";
import { TextReveal } from "../textReveal";
import { emptyTurnLog, type TurnLog } from "../turnLog";
import { newTurnSegment, type Segment } from "../turnTail";

const TICK_MS = 16;
let hidden = false;

beforeEach(() => {
  vi.useFakeTimers();
  hidden = false;
});

afterEach(() => {
  vi.useRealTimers();
});

function reveal(segments: () => Segment[], onTick = vi.fn()) {
  return new TextReveal(segments, onTick, {
    now: () => Date.now(),
    isHidden: () => hidden,
    tickMs: TICK_MS,
  });
}

it("re-renders only while the reveal moves", () => {
  const log = textLog("Let me fetch that page for you.");
  const segment = newTurnSegment("turn-1", log);
  const onTick = vi.fn();
  const text = reveal(() => [segment], onTick);
  text.track(emptyTurnLog(), log);

  vi.advanceTimersByTime(1_000);
  const shown = textOf(text.display(segment));
  const revealing = onTick.mock.calls.length;
  expect(revealing).toBeGreaterThan(0);

  // The block stays open; the last word waits for its end, so nothing moves.
  vi.advanceTimersByTime(1_000);
  expect(textOf(text.display(segment))).toBe(shown);
  expect(onTick).toHaveBeenCalledTimes(revealing);
  text.dispose();
});

it("flows a block sent in one delta over the lag cap, as the AI SDK path does", () => {
  const words = Array.from({ length: 200 }, (_, i) => `word${i}`).join(" ");
  const log = textLog(`${words} `);
  const segment = newTurnSegment("turn-1", log);
  const text = reveal(() => [segment]);
  text.track(emptyTurnLog(), log);

  vi.advanceTimersByTime(MAX_LAG_MS / 4);
  const early = textOf(text.display(segment))!.length;
  expect(early).toBeGreaterThan(0);
  expect(early).toBeLessThan(log.rows[0].content.length / 2);

  vi.advanceTimersByTime(MAX_LAG_MS);
  expect(textOf(text.display(segment))).toBe(log.rows[0].content);
  text.dispose();
});

it("reveals a slow model's text at the steady rate, as it arrives", () => {
  const segment = newTurnSegment("turn-1", textLog(""));
  const text = reveal(() => [segment]);
  // 20 characters every 100 ms: half the steady rate.
  const steadyMs = (20 * 1000) / STEADY_CHARS_PER_SECOND;
  for (let i = 0; i < 5; i++) {
    const next = textLog(segment.log.rows[0].content + "abcd ".repeat(4));
    text.track(segment.log, next);
    segment.log = next;
    vi.advanceTimersByTime(steadyMs + TICK_MS);
    expect(textOf(text.display(segment))).toBe(next.rows[0].content);
    vi.advanceTimersByTime(100 - steadyMs - TICK_MS);
  }
  text.dispose();
});

it("keeps revealing after the block ends, and reads as streaming until shown whole", () => {
  const open = textLog("One two three four five six seven eight nine ten");
  const segment = newTurnSegment("turn-1", open);
  const text = reveal(() => [segment]);
  text.track(emptyTurnLog(), open);
  segment.log = textLog(open.rows[0].content, false);

  vi.advanceTimersByTime(TICK_MS * 2);
  const midway = text.display(segment) as typeof segment;
  expect(midway.log.rows[0].content.length).toBeLessThan(
    open.rows[0].content.length,
  );
  expect(midway.log.blocks.b.open).toBe(true);

  vi.advanceTimersByTime(MAX_LAG_MS + TICK_MS);
  const done = text.display(segment) as typeof segment;
  expect(done.log.rows[0].content).toBe(open.rows[0].content);
  expect(done.log.blocks.b.open).toBe(false);
  text.dispose();
});

it("shows every complete word at once in a hidden tab", () => {
  hidden = true;
  const log = textLog("Lots of words arrive while the tab is hidden and");
  const segment = newTurnSegment("turn-1", log);
  const text = reveal(() => [segment]);
  text.track(emptyTurnLog(), log);
  vi.advanceTimersByTime(TICK_MS);
  expect(textOf(text.display(segment))).toBe(
    "Lots of words arrive while the tab is hidden ",
  );
  text.dispose();
});

it("cuts Japanese written without spaces into words instead of holding it back", () => {
  const log = textLog("今日はとても良い天気ですね。散歩に行きましょう。");
  const segment = newTurnSegment("turn-1", log);
  const text = reveal(() => [segment]);
  text.track(emptyTurnLog(), log);
  vi.advanceTimersByTime(MAX_LAG_MS / 2);
  const shown = textOf(text.display(segment))!.length;
  expect(shown).toBeGreaterThan(0);
  expect(shown).toBeLessThan(log.rows[0].content.length);
  text.dispose();
});

function textLog(content: string, open = true): TurnLog {
  return {
    ...emptyTurnLog(),
    turnId: "turn-1",
    rows: [
      {
        key: "block-1",
        role: "assistant",
        content,
        toolCalls: [],
        toolCallId: null,
        sequence: null,
        metadata: null,
      },
    ],
    blocks: { b: { kind: "text", open, row: 0 } },
  };
}

function textOf(segment: Segment) {
  return segment.kind === "turn" ? segment.log.rows[0].content : null;
}
