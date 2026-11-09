import { ApiError } from "@/lib/autogpt-server-api/helpers";
import type { UIMessage } from "ai";
import { describe, expect, it } from "vitest";
import {
  describeWait,
  hasAnswered,
  isEndedStage,
  isInterviewClosedError,
  STAGE_READ_INTERVAL_MS,
  STAGE_READ_WINDOW_MS,
  startFailureCopy,
  startFailureOf,
  withoutKickoff,
} from "../helpers";

function message(
  role: "user" | "assistant",
  text: string,
  metadata?: Record<string, unknown>,
): UIMessage {
  return {
    id: `${role}-${text}`,
    role,
    parts: [{ type: "text", text }],
    ...(metadata ? { metadata } : {}),
  };
}

const KICKOFF = message("user", "Start it now.", {
  hidden: true,
  kind: "onboarding_interview_kickoff",
});

describe("the interview's end", () => {
  it("reads the stage for ten seconds after a turn, every two", () => {
    expect(STAGE_READ_WINDOW_MS).toBe(10_000);
    expect(STAGE_READ_INTERVAL_MS).toBe(2_000);
  });

  it("is over once closed or handed off, and not before", () => {
    expect(isEndedStage("closed")).toBe(true);
    expect(isEndedStage("handed_off")).toBe(true);
    expect(isEndedStage("wrap_up_issued")).toBe(false);
    expect(isEndedStage(null)).toBe(false);
    expect(isEndedStage(undefined)).toBe(false);
  });

  it("knows a turn refused because the interview is over", () => {
    expect(
      isInterviewClosedError(
        new Error('{"detail":"onboarding_interview_closed"}'),
      ),
    ).toBe(true);
    expect(isInterviewClosedError(new Error("Network error"))).toBe(false);
    expect(isInterviewClosedError(undefined)).toBe(false);
  });
});

describe("what the person said", () => {
  it("leaves out the hidden kickoff the server wrote", () => {
    const greeting = message("assistant", "Hi, I'm Otto, an AI.");

    expect(withoutKickoff([KICKOFF, greeting])).toEqual([greeting]);
  });

  it("has answered only once a message of their own is in", () => {
    const greeting = message("assistant", "Hi, I'm Otto, an AI.");

    expect(hasAnswered([KICKOFF, greeting])).toBe(false);
    expect(hasAnswered([KICKOFF, greeting, message("user", "I bake.")])).toBe(
      true,
    );
  });
});

describe("a start that fails", () => {
  it("treats a 409 as an interview that is already over", () => {
    expect(startFailureOf(new ApiError("over", 409, null))).toBe("over");
  });

  it("reads the wait a spent budget sends", () => {
    const failure = startFailureOf(
      new ApiError("busy", 429, {
        detail: { message: "Slow down.", retry_after_seconds: 90 },
      }),
    );

    expect(failure).toEqual({ kind: "busy", retryAfterSeconds: 90 });
    expect(failure !== "over" && startFailureCopy(failure)).toBe(
      "You've sent a lot of messages in a short time. Your progress is saved; try again in 2 minutes.",
    );
  });

  it("says a little while when a 429 names no wait", () => {
    const failure = startFailureOf(
      new ApiError("busy", 429, { detail: "Too many chats at once." }),
    );

    expect(failure).toEqual({ kind: "busy", retryAfterSeconds: null });
    expect(failure !== "over" && startFailureCopy(failure)).toContain(
      "try again in a little while",
    );
  });

  it("tells an outage from any other failure", () => {
    const outage = startFailureOf(new ApiError("down", 503, null));
    const other = startFailureOf(new Error("socket hang up"));

    expect(outage).toEqual({ kind: "unavailable" });
    expect(outage !== "over" && startFailureCopy(outage)).toContain(
      "Otto can't chat right now",
    );
    expect(other).toEqual({ kind: "failed" });
    expect(other !== "over" && startFailureCopy(other)).toBe(
      "We couldn't start your interview. Try again.",
    );
  });

  it("rounds a wait up to the minute or the hour", () => {
    expect(describeWait(20)).toBe("a minute");
    expect(describeWait(61)).toBe("2 minutes");
    expect(describeWait(1200)).toBe("20 minutes");
    expect(describeWait(3600)).toBe("an hour");
    expect(describeWait(3601)).toBe("2 hours");
  });
});
