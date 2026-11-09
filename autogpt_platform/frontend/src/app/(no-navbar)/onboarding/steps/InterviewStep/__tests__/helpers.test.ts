import { ApiError } from "@/lib/autogpt-server-api/helpers";
import type { UIMessage } from "ai";
import { afterEach, describe, expect, it } from "vitest";
import {
  describeWait,
  finishFailureCopy,
  finishFailureOf,
  finishPathOf,
  forgetTyping,
  hasAnswered,
  isEndedStage,
  isInterviewClosedError,
  isTypingRemembered,
  refusalOf,
  rememberTyping,
  STAGE_READ_INTERVAL_MS,
  STAGE_READ_WINDOW_MS,
  startFailureCopy,
  startFailureOf,
  turnRefusalOf,
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
  it("treats a 409 as a plain failure to try again", () => {
    expect(startFailureOf(new ApiError("conflict", 409, null))).toEqual({
      kind: "failed",
    });
  });

  it("reads the wait a spent budget sends", () => {
    const failure = startFailureOf(
      new ApiError("busy", 429, {
        detail: { message: "Slow down.", retry_after_seconds: 90 },
      }),
    );

    expect(failure).toEqual({ kind: "busy", retryAfterSeconds: 90 });
    expect(startFailureCopy(failure)).toBe(
      "You've sent a lot of messages in a short time. Your progress is saved; try again in 2 minutes.",
    );
  });

  it("says a little while when a 429 names no wait", () => {
    const failure = startFailureOf(
      new ApiError("busy", 429, { detail: "Too many chats at once." }),
    );

    expect(failure).toEqual({ kind: "busy", retryAfterSeconds: null });
    expect(startFailureCopy(failure)).toContain("try again in a little while");
  });

  it("tells an outage from any other failure", () => {
    const outage = startFailureOf(new ApiError("down", 503, null));
    const other = startFailureOf(new Error("socket hang up"));

    expect(outage).toEqual({ kind: "unavailable" });
    expect(startFailureCopy(outage)).toContain("Otto can't chat right now");
    expect(other).toEqual({ kind: "failed" });
    expect(startFailureCopy(other)).toBe(
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

describe("what a finish answered", () => {
  function take(status: "transcribed" | "failed", errorCode?: string | null) {
    return { status, input_mode: "typed" as const, error_code: errorCode };
  }

  it("moves on as path A for a take that went through or a newer take", () => {
    expect(finishPathOf(take("transcribed"))).toBe("A");
    expect(finishPathOf(take("failed", "superseded"))).toBe("A");
  });

  it("moves on as path B for a verdict on the words", () => {
    expect(finishPathOf(take("failed", "no_usable_speech"))).toBe("B");
    expect(finishPathOf(take("failed", "insufficient_content"))).toBe("B");
  });

  it("stays for a failure nobody ruled on", () => {
    expect(finishPathOf(take("failed", "quality_check_unavailable"))).toBe(
      null,
    );
    expect(finishPathOf(take("failed", "something_else"))).toBe(null);
    expect(finishPathOf(take("failed", null))).toBe(null);
  });

  it("says how long to wait when the budget turns it away, and try again otherwise", () => {
    const busy = finishFailureOf(
      new ApiError("busy", 429, {
        detail: { message: "Slow down.", retry_after_seconds: 1200 },
      }),
    );
    const down = finishFailureOf(new ApiError("down", 503, null));

    expect(finishFailureCopy(busy)).toBe(
      "You've sent a lot of messages in a short time. Your progress is saved; try again in 20 minutes.",
    );
    expect(finishFailureCopy(down)).toBe("That didn't go through. Try again.");
    expect(finishFailureCopy(finishFailureOf(new Error("offline")))).toBe(
      "That didn't go through. Try again.",
    );
  });
});

describe("a refusal because a voice call holds the interview", () => {
  const CALL_ID = "e".repeat(32);

  it("names the live call a start was refused for", () => {
    const refusal = refusalOf(
      new ApiError("held", 409, {
        detail: { code: "call_in_progress", call_id: CALL_ID },
      }),
    );

    expect(refusal).toEqual({
      status: 409,
      retryAfterSeconds: null,
      heldBy: { kind: "call_in_progress", callId: CALL_ID },
    });
  });

  it("names the call that took the interview, or none when the typed interview is over", () => {
    expect(
      refusalOf(
        new ApiError("closed", 409, {
          detail: { code: "onboarding_interview_closed", call_id: CALL_ID },
        }),
      ).heldBy,
    ).toEqual({ kind: "closed", callId: CALL_ID });
    expect(
      refusalOf(
        new ApiError("closed", 409, {
          detail: { code: "onboarding_interview_closed" },
        }),
      ).heldBy,
    ).toEqual({ kind: "closed", callId: null });
  });

  it("finds no call in any other refusal", () => {
    expect(
      refusalOf(new ApiError("busy", 409, { detail: "turn_in_progress" }))
        .heldBy,
    ).toBeNull();
    expect(
      refusalOf(
        new ApiError("held", 409, { detail: { code: "call_in_progress" } }),
      ).heldBy,
    ).toBeNull();
    expect(refusalOf(new Error("offline"))).toEqual({
      status: null,
      retryAfterSeconds: null,
      heldBy: null,
    });
  });

  it("reads the call a refused typed turn names from the response the chat reports", () => {
    const refused = new Error(
      JSON.stringify({
        detail: { code: "onboarding_interview_closed", call_id: CALL_ID },
      }),
    );

    expect(turnRefusalOf(refused)).toEqual({ kind: "closed", callId: CALL_ID });
  });

  it("leaves a typed turn refused for the typed interview's own close to the chat", () => {
    expect(
      turnRefusalOf(
        new Error(JSON.stringify({ detail: "onboarding_interview_closed" })),
      ),
    ).toBeNull();
    expect(
      turnRefusalOf(
        new Error(
          JSON.stringify({ detail: { code: "onboarding_interview_closed" } }),
        ),
      ),
    ).toBeNull();
    expect(turnRefusalOf(new Error("Failed to fetch"))).toBeNull();
    expect(turnRefusalOf(new Error("null"))).toBeNull();
    expect(turnRefusalOf(undefined)).toBeNull();
  });
});

describe("a choice to type, kept for this tab", () => {
  afterEach(() => sessionStorage.clear());

  it("is kept per person until it is forgotten", () => {
    rememberTyping("typist");

    expect(isTypingRemembered("typist")).toBe(true);
    expect(isTypingRemembered("someone-else")).toBe(false);

    forgetTyping("typist");
    expect(isTypingRemembered("typist")).toBe(false);
  });

  it("keeps nothing for nobody", () => {
    rememberTyping(null);
    forgetTyping(null);

    expect(isTypingRemembered(null)).toBe(false);
    expect(sessionStorage.length).toBe(0);
  });
});
