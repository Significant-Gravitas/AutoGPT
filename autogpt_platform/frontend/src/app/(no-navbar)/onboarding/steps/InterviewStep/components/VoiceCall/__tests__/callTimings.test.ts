import { describe, expect, it, vi } from "vitest";
import { type CallMarks, reportCallStart, startSeconds } from "../callTimings";

interface SentSpan {
  options: {
    name: string;
    op?: string;
    startTime?: number;
    parentSpan?: unknown;
  };
  endedAt: number | undefined;
}

const sentry = vi.hoisted(() => {
  const spans: SentSpan[] = [];
  return {
    spans,
    startInactiveSpan: (options: SentSpan["options"]) => {
      const span: SentSpan & { end: (at?: number) => void } = {
        options,
        endedAt: undefined,
        end(at?: number) {
          span.endedAt = at;
        },
      };
      spans.push(span);
      return span;
    },
  };
});

vi.mock("@sentry/nextjs", () => ({
  startInactiveSpan: sentry.startInactiveSpan,
}));

const GRANTED = 1_700_000_000_000;

function marks(overrides: Partial<CallMarks> = {}): CallMarks {
  return {
    granted: GRANTED,
    minted: GRANTED + 400,
    joined: GRANTED + 1_200,
    ready: GRANTED + 2_000,
    firstSound: GRANTED + 2_900,
    ...overrides,
  };
}

describe("how long a call took to start", () => {
  it("counts each step reached from the microphone being granted", () => {
    expect(startSeconds(marks())).toEqual({
      mint_s: 0.4,
      room_s: 1.2,
      ready_s: 2,
    });
    expect(startSeconds(marks({ joined: null, ready: null }))).toEqual({
      mint_s: 0.4,
      room_s: null,
      ready_s: null,
    });
  });

  it("sends Sentry one span from the microphone to Otto's first sound, with a span for each step", () => {
    sentry.spans.length = 0;

    reportCallStart(marks());

    const [start, ...steps] = sentry.spans;
    expect(start.options).toMatchObject({
      name: "Voice call start",
      op: "voice.call.start",
      startTime: GRANTED,
      parentSpan: null,
    });
    expect(start.endedAt).toBe(GRANTED + 2_900);
    expect(
      steps.map(({ options, endedAt }) => [
        options.op,
        options.startTime,
        endedAt,
        options.parentSpan === start,
      ]),
    ).toEqual([
      ["voice.call.mint", GRANTED, GRANTED + 400, true],
      ["voice.call.room", GRANTED + 400, GRANTED + 1_200, true],
      ["voice.call.ready", GRANTED + 1_200, GRANTED + 2_000, true],
      ["voice.call.first_sound", GRANTED + 2_000, GRANTED + 2_900, true],
    ]);
  });

  it.each([["minted"], ["joined"], ["ready"], ["firstSound"]] as const)(
    "sends nothing for a start that never reached %s",
    (step) => {
      sentry.spans.length = 0;

      reportCallStart(marks({ [step]: null }));

      expect(sentry.spans).toEqual([]);
    },
  );
});
