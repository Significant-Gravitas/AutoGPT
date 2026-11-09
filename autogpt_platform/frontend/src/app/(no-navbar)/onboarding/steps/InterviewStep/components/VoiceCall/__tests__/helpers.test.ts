import type { VoiceCallStatusResponse } from "@/app/api/__generated__/models/voiceCallStatusResponse";
import { describe, expect, it } from "vitest";
import {
  answerOf,
  callStartFailureCopy,
  callStartFailureOf,
  callStatusCopy,
  ENDING_POLL_MS,
  ENDING_WINDOW_MS,
  endedCopy,
  interviewOf,
  isRelaySocketUrl,
  microphoneRefusalOf,
  parseRelayMessage,
  REATTACH_WINDOW_MS,
  ranToClose,
  secondsBetween,
  timeLeftCopy,
  turnsLeftCopy,
  voiceFinishOutcome,
  withCaption,
} from "../helpers";

const CALL_ID = "c".repeat(32);
const ACCEPTED = { status: "transcribed", input_mode: "typed" } as const;

function call(
  overrides: Partial<VoiceCallStatusResponse> & {
    interview?: "taken" | "superseded" | null;
  } = {},
): VoiceCallStatusResponse {
  return {
    call_id: CALL_ID,
    status: "handed_off",
    end_reason: "finished",
    leg: "websocket",
    hand_off: ACCEPTED,
    ...overrides,
  } as VoiceCallStatusResponse;
}

function failed(errorCode: string | null) {
  return {
    status: "failed",
    input_mode: "typed",
    error_code: errorCode,
  } as const;
}

describe("the relay's messages", () => {
  it("reads every message the relay sends", () => {
    expect(
      parseRelayMessage(
        JSON.stringify({ type: "turn_complete", turn: 2, turns_left: 13 }),
      ),
    ).toEqual({ type: "turn_complete", turn: 2, turns_left: 13 });
  });

  it("keeps only the fields each message carries", () => {
    expect(
      parseRelayMessage(
        JSON.stringify({ type: "agent_text", text: "Hi.", note: "extra" }),
      ),
    ).toEqual({ type: "agent_text", text: "Hi." });
    expect(
      parseRelayMessage(
        JSON.stringify({
          type: "error",
          error: "call_in_progress",
          call_id: 7,
        }),
      ),
    ).toEqual({ type: "error", error: "call_in_progress", call_id: null });
  });

  it.each([
    ["text that is not JSON", "not json"],
    ["a JSON value that is not an object", "42"],
    ["a message type the relay never sends", '{"type":"note","text":"hi"}'],
    ["a message with no type", '{"text":"hi"}'],
    [
      "a ready with no caps",
      '{"type":"ready","call_id":"c","resume":"t","leg":"websocket"}',
    ],
    [
      "a ready whose caps are not numbers",
      '{"type":"ready","call_id":"c","resume":"t","leg":"websocket","caps":{"max_call_s":"300"}}',
    ],
    [
      "a ready on a leg the relay has none of",
      '{"type":"ready","call_id":"c","resume":"t","leg":"carrier","caps":{}}',
    ],
    ["a caption with no text", '{"type":"agent_text"}'],
    ["a caption whose text is not text", '{"type":"user_text","text":{"a":1}}'],
    [
      "a turn count that is not a number",
      '{"type":"turn_complete","turn":1,"turns_left":"many"}',
    ],
    ["an ended with no elapsed time", '{"type":"ended","reason":"finished"}'],
    ["a resumed with no gap", '{"type":"resumed"}'],
    ["an error with no error", '{"type":"error","call_id":"c"}'],
  ])("ignores %s", (_case, data) => {
    expect(parseRelayMessage(data)).toBeNull();
  });
});

describe("the relay socket the server names", () => {
  it("is a secure socket", () => {
    expect(isRelaySocketUrl("wss://api.example.com/voice/call")).toBe(true);
  });

  it("is a plain socket only on a page served over plain HTTP", () => {
    // The test page is served from http://localhost.
    expect(isRelaySocketUrl("ws://localhost:8010/voice/call")).toBe(
      window.location.protocol === "http:",
    );
  });

  it.each([
    ["a page address", "https://api.example.com/voice/call"],
    ["no address at all", "not a url"],
    ["something other than text", 42],
  ])("is never %s", (_case, value) => {
    expect(isRelaySocketUrl(value)).toBe(false);
  });
});

describe("what a call's record answers", () => {
  it("waits while the call runs or its words are on their way to a take", () => {
    // `handing_off` is how the call routes report a call owed a hand-off.
    for (const status of [
      "connecting",
      "live",
      "ended",
      "transcribing",
      "handing_off",
    ]) {
      expect(
        answerOf(
          call({
            status: status as VoiceCallStatusResponse["status"],
            hand_off: null,
          }),
        ),
      ).toBeNull();
    }
  });

  it.each([
    ["finished"],
    ["wall_clock"],
    ["turn_cap"],
    ["spend_cap"],
    ["scope"],
    ["user_hangup"],
  ])(
    "leads a call that ran to its close (%s) straight on to the team",
    (reason) => {
      expect(answerOf(call({ end_reason: reason }))).toEqual({
        kind: "team",
        path: "A",
      });
    },
  );

  it.each([
    ["client_gone"],
    ["idle"],
    ["vendor_error"],
    ["server_restart"],
    ["superseded"],
    ["crashed"],
  ])("shows the lost screen for a call cut off (%s)", (reason) => {
    expect(answerOf(call({ end_reason: reason }))).toEqual({
      kind: "lost",
      reason,
    });
  });

  it("opens the typed interview for a call a later take superseded, whatever else it says", () => {
    expect(answerOf(call({ interview: "superseded" }))).toBe("typed");
    expect(
      answerOf(
        call({
          status: "transcribing",
          hand_off: null,
          interview: "superseded",
        }),
      ),
    ).toBe("typed");
  });

  it("reads the record's word that a cut-off call took the interview before its end reason", () => {
    expect(
      answerOf(call({ end_reason: "client_gone", interview: "taken" })),
    ).toEqual({ kind: "team", path: "A" });
  });

  it("treats a call the server named as the interview's take as taken", () => {
    expect(answerOf(call({ end_reason: "idle" }), { taken: true })).toEqual({
      kind: "team",
      path: "A",
    });
  });

  it("finishes a closed call whose confirm is still awaited before moving on", () => {
    expect(answerOf(call({ status: "transcribed", hand_off: null }))).toEqual({
      kind: "team",
      path: null,
    });
    expect(
      answerOf(
        call({ status: "transcribed", hand_off: null, end_reason: "idle" }),
      ),
    ).toEqual({ kind: "lost", reason: "idle" });
  });

  it.each([["no_usable_speech"], ["insufficient_content"]])(
    "shows the insufficient screen for a verdict on the words (%s), however the call ended",
    (code) => {
      for (const reason of ["finished", "client_gone"]) {
        expect(
          answerOf(
            call({
              status: "failed",
              end_reason: reason,
              hand_off: failed(code),
            }),
          ),
        ).toEqual({ kind: "insufficient" });
      }
    },
  );

  it("offers Try again for a hand-off the pipeline never ruled on, with the other ways on after a cut-off call", () => {
    const unruled = failed("quality_check_unavailable");
    expect(answerOf(call({ status: "failed", hand_off: unruled }))).toEqual({
      kind: "retry",
      cutOff: false,
      reason: "finished",
    });
    expect(
      answerOf(
        call({
          status: "failed",
          end_reason: "vendor_error",
          hand_off: unruled,
        }),
      ),
    ).toEqual({ kind: "retry", cutOff: true, reason: "vendor_error" });
  });

  it.each([
    ["transcription_unavailable"],
    ["recording_unavailable"],
    ["some_new_failure"],
  ])(
    "offers another call or typing for a call that kept nothing to hand off (%s)",
    (code) => {
      for (const reason of ["finished", "client_gone"]) {
        expect(
          answerOf(
            call({
              status: "failed",
              end_reason: reason,
              hand_off: failed(code),
            }),
          ),
        ).toEqual({ kind: "unsaved", reason });
      }
    },
  );

  it.each([
    ["quality_check_unavailable"],
    ["rate_limited"],
    ["personalization_unavailable"],
    [null],
  ])(
    "offers Try again only for words the call kept and nobody ruled on (%s)",
    (code) => {
      expect(
        answerOf(call({ status: "failed", hand_off: failed(code) })),
      ).toEqual({
        kind: "retry",
        cutOff: false,
        reason: "finished",
      });
    },
  );

  it("knows only the two words the record can say on the interview", () => {
    expect(interviewOf(call({ interview: "taken" }))).toBe("taken");
    expect(interviewOf(call({ interview: "superseded" }))).toBe("superseded");
    expect(interviewOf(call({ interview: null }))).toBeNull();
    expect(interviewOf(call())).toBeNull();
  });

  it("tells a call that ran to its close from one cut off", () => {
    expect(ranToClose("user_hangup")).toBe(true);
    expect(ranToClose("client_gone")).toBe(false);
    expect(ranToClose(null)).toBe(false);
  });
});

describe("what a finish of a call answers", () => {
  it("leads on for an accepted take, or one a newer take owns", () => {
    expect(voiceFinishOutcome(ACCEPTED)).toBe("A");
    expect(voiceFinishOutcome(failed("superseded"))).toBe("A");
  });

  it("keeps the interview open for a verdict on the words", () => {
    expect(voiceFinishOutcome(failed("insufficient_content"))).toBe(
      "insufficient",
    );
  });

  it("asks to try again when the pipeline never ruled", () => {
    expect(voiceFinishOutcome(failed("rate_limited"))).toBeNull();
    expect(voiceFinishOutcome(failed(null))).toBeNull();
  });

  it("says when the call kept nothing to hand off", () => {
    expect(voiceFinishOutcome(failed("transcription_unavailable"))).toBe(
      "unsaved",
    );
  });
});

describe("the call's copy", () => {
  it("says what ended the call, and something plain for a reason it does not know", () => {
    expect(endedCopy("idle")).toBe(
      "We didn't hear anything for a while, so the call ended.",
    );
    expect(endedCopy("something_new")).toBe("The call ended.");
    // Ended through the end route, by the person or from another device.
    expect(endedCopy("superseded")).toBe(
      "The call ended before Otto could finish.",
    );
    expect(endedCopy(null)).toBe("The call ended.");
  });

  it("never mentions typing while a lost call's words are being saved", () => {
    const saving = callStatusCopy({
      phase: "saving",
      isWrappingUp: false,
      wasResumed: false,
      isHangingUp: false,
    });
    expect(saving).toBe("We lost the call. Saving what you said…");
    expect(saving).not.toMatch(/typ/i);
  });

  it("puts the live call's state in one line", () => {
    const live = { phase: "live" as const, wasResumed: false };
    expect(
      callStatusCopy({ ...live, isWrappingUp: false, isHangingUp: true }),
    ).toBe("Hanging up…");
    expect(
      callStatusCopy({ ...live, isWrappingUp: true, isHangingUp: false }),
    ).toBe("Otto is wrapping up");
    expect(
      callStatusCopy({
        phase: "ended",
        isWrappingUp: false,
        wasResumed: false,
        isHangingUp: false,
      }),
    ).toBeNull();
  });

  it("counts a call's length to a tenth of a second", () => {
    expect(secondsBetween(1_000, 93_440)).toBe(92.4);
    expect(secondsBetween(5_000, 1_000)).toBe(0);
  });

  it("counts the time and the turns left", () => {
    expect(timeLeftCopy(299_400)).toBe("4:59 left");
    expect(timeLeftCopy(-5_000)).toBe("0:00 left");
    expect(turnsLeftCopy(2)).toBe("2 turns left");
    expect(turnsLeftCopy(1)).toBe("1 turn left");
    expect(turnsLeftCopy(0)).toBe("Wrapping up");
  });

  it("explains a call that could not start, and when another can", () => {
    expect(
      callStartFailureCopy(
        callStartFailureOf({
          status: 429,
          retryAfterSeconds: 3_600,
          heldBy: null,
        }),
      ),
    ).toBe(
      "You've made a few calls already. You can call again in an hour, or type instead.",
    );
    expect(
      callStartFailureCopy({ kind: "busy", retryAfterSeconds: null }),
    ).toBe(
      "You've made a few calls already. You can call again a little later, or type instead.",
    );
    expect(
      callStartFailureCopy(
        callStartFailureOf({
          status: 503,
          retryAfterSeconds: 30,
          heldBy: null,
        }),
      ),
    ).toBe(
      "Otto can't take calls right now. You can type instead, or try again in a moment.",
    );
    expect(
      callStartFailureCopy(
        callStartFailureOf({
          status: null,
          retryAfterSeconds: null,
          heldBy: null,
        }),
      ),
    ).toBe("We couldn't start the call. Try again, or type instead.");
  });
});

describe("a refused microphone", () => {
  it("tells the browser's refusal from a device that cannot open", () => {
    expect(microphoneRefusalOf({ name: "NotAllowedError" })).toBe("blocked");
    expect(microphoneRefusalOf({ name: "SecurityError" })).toBe("blocked");
    expect(microphoneRefusalOf({ name: "NotFoundError" })).toBe("unavailable");
    expect(microphoneRefusalOf("no name at all")).toBe("unavailable");
  });
});

describe("the captions", () => {
  it("runs one speaker's fragments into one line until the turn ends", () => {
    let lines = withCaption([], "otto", "Hi, I'm Otto.", true);
    lines = withCaption(lines, "otto", " What do you do?", false);
    lines = withCaption(lines, "otto", "Tell me more.", true);
    lines = withCaption(lines, "you", " I bake.", false);

    expect(lines.map(({ who, text }) => ({ who, text }))).toEqual([
      { who: "otto", text: "Hi, I'm Otto. What do you do?" },
      { who: "otto", text: "Tell me more." },
      { who: "you", text: "I bake." },
    ]);
    expect(new Set(lines.map((line) => line.id)).size).toBe(3);
  });
});

describe("the call's waits", () => {
  it("holds a dropped call for the relay's re-attach window, and reads its words for thirty seconds every two", () => {
    expect(REATTACH_WINDOW_MS).toBe(15_000);
    expect(ENDING_WINDOW_MS).toBe(30_000);
    expect(ENDING_POLL_MS).toBe(2_000);
  });
});
