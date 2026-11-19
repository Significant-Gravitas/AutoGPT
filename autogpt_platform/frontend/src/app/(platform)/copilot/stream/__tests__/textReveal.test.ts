import { afterEach, beforeEach, expect, it, vi } from "vitest";

import { TextReveal } from "../textReveal";
import { emptyTurnLog, type TurnLog } from "../turnLog";
import { newTurnSegment, type Segment } from "../turnTail";

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

it("re-renders only while the reveal moves", () => {
  const log = openTextLog("Let me fetch that page for you.");
  const segment = newTurnSegment("turn-1", log);
  const onTick = vi.fn();
  const reveal = new TextReveal(() => [segment], onTick);
  reveal.track(emptyTurnLog(), log);

  vi.advanceTimersByTime(1_000);
  const shown = textOf(reveal.display(segment));
  const revealing = onTick.mock.calls.length;
  expect(revealing).toBeGreaterThan(0);

  // The block stays open; the last word waits for its end, so nothing moves.
  vi.advanceTimersByTime(1_000);
  expect(textOf(reveal.display(segment))).toBe(shown);
  expect(onTick).toHaveBeenCalledTimes(revealing);
  reveal.dispose();
});

function openTextLog(content: string): TurnLog {
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
    blocks: { "block-1": { kind: "text", open: true, row: 0 } },
  };
}

function textOf(segment: Segment) {
  return segment.kind === "turn" ? segment.log.rows[0].content : null;
}
