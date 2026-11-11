import { describe, expect, it, vi } from "vitest";
import { type CallMarks, reportCallStart, startSeconds } from "../callTimings";

interface SentSpan {
  options: {
    name: string;
    op?: string;
    startTime?: number;
    parentSpan?: unknown;
    attributes?: Record<string, unknown>;
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
const CALL_ID = "a".repeat(32);

function marks(overrides: Partial<CallMarks> = {}): CallMarks {
  return {
    clicked: GRANTED - 1_500,
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

  it("sends Sentry one span from the click to Otto's first sound, with a span for each step", () => {
    sentry.spans.length = 0;

    reportCallStart(CALL_ID, marks());

    const [start, ...steps] = sentry.spans;
    expect(start.options).toMatchObject({
      name: "Voice call start",
      op: "voice.call.start",
      startTime: GRANTED - 1_500,
      parentSpan: null,
      attributes: {
        call_id: CALL_ID,
        granted_s: 1.5,
        mint_s: 0.4,
        room_s: 1.2,
        ready_s: 2,
        first_sound_s: 2.9,
      },
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
      ["voice.call.grant", GRANTED - 1_500, GRANTED, true],
      ["voice.call.mint", GRANTED, GRANTED + 400, true],
      ["voice.call.room", GRANTED + 400, GRANTED + 1_200, true],
      ["voice.call.ready", GRANTED + 1_200, GRANTED + 2_000, true],
      ["voice.call.first_sound", GRANTED + 2_000, GRANTED + 2_900, true],
    ]);
  });

  it("ends at the relay's ready when Otto was heard before it reached the page", () => {
    sentry.spans.length = 0;

    reportCallStart(CALL_ID, marks({ firstSound: GRANTED + 1_800 }));

    const [start, ...steps] = sentry.spans;
    expect(start.options.attributes).toMatchObject({
      ready_s: 2,
      first_sound_s: 1.8,
    });
    expect(start.endedAt).toBe(GRANTED + 2_000);
    // No step lasts less than no time.
    expect(
      steps.map(({ options, endedAt }) => [
        options.op,
        (endedAt ?? 0) - (options.startTime ?? 0),
      ]),
    ).toEqual([
      ["voice.call.grant", 1_500],
      ["voice.call.mint", 400],
      ["voice.call.room", 800],
      ["voice.call.ready", 800],
      ["voice.call.first_sound", 0],
    ]);
  });

  it("ends at the page's room join when the relay was ready, and Otto heard, before it settled", () => {
    sentry.spans.length = 0;

    reportCallStart(
      CALL_ID,
      marks({ joined: GRANTED + 2_500, firstSound: GRANTED + 2_400 }),
    );

    const [start, ...steps] = sentry.spans;
    expect(start.endedAt).toBe(GRANTED + 2_500);
    expect(
      steps.map(({ options, endedAt }) => [
        options.op,
        (endedAt ?? 0) - (options.startTime ?? 0),
      ]),
    ).toEqual([
      ["voice.call.grant", 1_500],
      ["voice.call.mint", 400],
      ["voice.call.room", 2_100],
      ["voice.call.ready", 0],
      ["voice.call.first_sound", 400],
    ]);
  });

  it.each([["minted"], ["joined"], ["ready"], ["firstSound"]] as const)(
    "sends nothing for a start that never reached %s",
    (step) => {
      sentry.spans.length = 0;

      reportCallStart(CALL_ID, marks({ [step]: null }));

      expect(sentry.spans).toEqual([]);
    },
  );
});
