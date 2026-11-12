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
  isRelaySocketUrl,
  microphoneRefusalOf,
  parseRelayMessage,
  REATTACH_WINDOW_MS,
  ROOM_JOIN_TIMEOUT_MS,
  RoomJoinTimeout,
  ranToClose,
  roomJoinFailureOf,
  roomPassOf,
  secondsBetween,
  timeLeftCopy,
  turnsLeftCopy,
  voiceFinishOutcome,
  withCaption,
} from "../helpers";

const CALL_ID = "c".repeat(32);
const ACCEPTED = { status: "transcribed", input_mode: "typed" } as const;

function call(
  overrides: Partial<VoiceCallStatusResponse> = {},
): VoiceCallStatusResponse {
  return {
    call_id: CALL_ID,
    status: "handed_off",
    end_reason: "finished",
    hand_off: ACCEPTED,
    ...overrides,
  };
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

  it.each(["probe", "captions"] as const)(
    "reads the relay finding Otto's voice in the microphone by its %s",
    (by) => {
      expect(
        parseRelayMessage(JSON.stringify({ type: "duplex", mode: "half", by })),
      ).toEqual({ type: "duplex", mode: "half", by });
    },
  );

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
    expect(
      parseRelayMessage(
        JSON.stringify({
          type: "ready",
          call_id: "c",
          caps: {
            max_call_s: 300,
            soft_wrap_s: 30,
            max_agent_turns: 16,
            soft_wrap_turns: 2,
            idle_s: 20,
          },
          resume: "t",
          leg: "websocket",
        }),
      ),
    ).not.toHaveProperty("leg");
  });

  it.each([
    ["text that is not JSON", "not json"],
    ["a JSON value that is not an object", "42"],
    ["a message type the relay never sends", '{"type":"note","text":"hi"}'],
    ["a message with no type", '{"text":"hi"}'],
    ["a ready with no caps", '{"type":"ready","call_id":"c","resume":"t"}'],
    [
      "a ready whose caps are not numbers",
      '{"type":"ready","call_id":"c","resume":"t","caps":{"max_call_s":"300"}}',
    ],
    [
      "a leg message, which the relay no longer sends",
      '{"type":"leg","leg":"websocket"}',
    ],
    ["a caption with no text", '{"type":"agent_text"}'],
    [
      "a duplex mode the relay never sends",
      '{"type":"duplex","mode":"full","by":"probe"}',
    ],
    [
      "a duplex found by nothing the relay names",
      '{"type":"duplex","mode":"half","by":"guess"}',
    ],
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
    const unsettled: VoiceCallStatusResponse["status"][] = [
      "connecting",
      "live",
      "ended",
      "transcribing",
      "handing_off",
    ];
    for (const status of unsettled) {
      expect(answerOf(call({ status, hand_off: null }))).toBeNull();
    }
  });

  it.each([
    ["finished"],
    ["wall_clock"],
    ["turn_cap"],
    ["spend_cap"],
    ["scope"],
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
    ["user_hangup"],
    ["client_gone"],
    ["idle"],
    ["vendor_error"],
    ["server_restart"],
    ["superseded"],
    ["crashed"],
  ])("shows the lost screen for a call hung up or cut off (%s)", (reason) => {
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

  it("shows the insufficient screen when the words were too few, however the call ended", () => {
    for (const reason of ["finished", "client_gone"]) {
      expect(
        answerOf(
          call({
            status: "failed",
            end_reason: reason,
            hand_off: failed("insufficient_content"),
          }),
        ),
      ).toEqual({ kind: "insufficient" });
    }
  });

  it("says it did not hear the person on a call where nothing was heard, however it ended", () => {
    for (const reason of ["finished", "client_gone"]) {
      expect(
        answerOf(
          call({
            status: "failed",
            end_reason: reason,
            hand_off: failed("no_usable_speech"),
          }),
        ),
      ).toEqual({ kind: "no_words" });
    }
  });

  it("offers Try again for a hand-off the pipeline never ruled on, with the other ways on after a call hung up or cut off", () => {
    const unruled = failed("quality_check_unavailable");
    expect(answerOf(call({ status: "failed", hand_off: unruled }))).toEqual({
      kind: "retry",
      leftOpen: false,
      reason: "finished",
    });
    for (const reason of ["user_hangup", "vendor_error"]) {
      expect(
        answerOf(
          call({ status: "failed", end_reason: reason, hand_off: unruled }),
        ),
      ).toEqual({ kind: "retry", leftOpen: true, reason });
    }
  });

  it.each([
    ["transcription_unavailable"],
    ["recording_unavailable"],
    ["some_new_failure"],
  ])(
    "offers another call or typing after a failure of ours saving the call (%s), never as though nothing was heard",
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
        leftOpen: false,
        reason: "finished",
      });
    },
  );

  it("leads on as path A from a closed call whose take another take beat", () => {
    expect(answerOf(call({ hand_off: failed("superseded") }))).toEqual({
      kind: "team",
      path: "A",
    });
  });

  it("tells a call that ran to its close from one hung up or cut off", () => {
    expect(ranToClose("finished")).toBe(true);
    expect(ranToClose("user_hangup")).toBe(false);
    expect(ranToClose("client_gone")).toBe(false);
    expect(ranToClose(null)).toBe(false);
  });
});

describe("what a finish of a call answers", () => {
  it("leads on for an accepted take", () => {
    expect(voiceFinishOutcome(ACCEPTED)).toBe("A");
  });

  it("says when another take owns the interview, for the call's record to tell whose", () => {
    expect(voiceFinishOutcome(failed("superseded"))).toBe("superseded");
  });

  it("keeps the interview open for a call where nothing was heard, or too little", () => {
    expect(voiceFinishOutcome(failed("insufficient_content"))).toBe(
      "insufficient",
    );
    expect(voiceFinishOutcome(failed("no_usable_speech"))).toBe("no_words");
  });

  it("asks to try again when the pipeline never ruled", () => {
    expect(voiceFinishOutcome(failed("rate_limited"))).toBeNull();
    expect(voiceFinishOutcome(failed(null))).toBeNull();
  });

  it("says when a failure of ours kept nothing of the call to hand off", () => {
    for (const code of [
      "transcription_unavailable",
      "recording_unavailable",
      "some_new_failure",
    ]) {
      expect(voiceFinishOutcome(failed(code))).toBe("unsaved");
    }
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
    // The relay lost the call's room.
    expect(endedCopy("room_error")).toBe("The call's audio dropped.");
  });

  it("says only that Otto wrapped up a finished call whose hand-off failed", () => {
    expect(endedCopy("finished")).toBe("That's everything Otto needed.");
    expect(endedCopy("finished", { handOffFailed: true })).toBe(
      "Otto wrapped up the call.",
    );
    // Any other end says what ended the call either way.
    expect(endedCopy("user_hangup", { handOffFailed: true })).toBe(
      "You ended the call.",
    );
  });

  it("never mentions typing while a lost call's words are being saved", () => {
    const saving = callStatusCopy({
      phase: "saving",
      isWrappingUp: false,
      wasResumed: false,
      isHangingUp: false,
      isAudioReconnecting: false,
    });
    expect(saving).toBe("We lost the call. Saving what you said…");
    expect(saving).not.toMatch(/typ/i);
  });

  it("says the words are being saved, not that anything failed, while a finish waits", () => {
    expect(
      callStatusCopy({
        phase: "finishing",
        isWrappingUp: false,
        wasResumed: false,
        isHangingUp: false,
        isAudioReconnecting: false,
      }),
    ).toBe("Saving what you said…");
  });

  it("puts the live call's state in one line", () => {
    const live = {
      phase: "live" as const,
      wasResumed: false,
      isAudioReconnecting: false,
    };
    expect(
      callStatusCopy({ ...live, isWrappingUp: false, isHangingUp: true }),
    ).toBe("Hanging up…");
    expect(
      callStatusCopy({ ...live, isWrappingUp: true, isHangingUp: false }),
    ).toBe("Otto is wrapping up");
    // LiveKit getting the audio back says so over the rest, Hang up aside.
    expect(
      callStatusCopy({
        ...live,
        isWrappingUp: true,
        isHangingUp: false,
        wasResumed: true,
        isAudioReconnecting: true,
      }),
    ).toBe("Reconnecting…");
    expect(
      callStatusCopy({
        phase: "ended",
        isWrappingUp: false,
        wasResumed: false,
        isHangingUp: false,
        isAudioReconnecting: false,
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
    expect(callStartFailureCopy({ kind: "audio" })).toBe(
      "We couldn't connect the call's audio. Try again, or type instead.",
    );
    expect(callStartFailureCopy({ kind: "unsupported" })).toBe(
      "This browser can't make calls. You can type instead, or call from another browser.",
    );
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

describe("the call's room", () => {
  const ROOM = "wss://agpt.livekit.cloud";

  it("joins the room the start names", () => {
    expect(roomPassOf({ url: ROOM, token: "jwt" })).toEqual({
      url: ROOM,
      token: "jwt",
    });
  });

  it("joins no room a start names without a secure socket or a token", () => {
    expect(
      roomPassOf({ url: "https://agpt.livekit.cloud", token: "jwt" }),
    ).toBeNull();
    expect(roomPassOf({ url: ROOM, token: "" })).toBeNull();
  });

  it("gives the join ten seconds, for a network that is slow to let it through", () => {
    expect(ROOM_JOIN_TIMEOUT_MS).toBe(10_000);
  });

  it("says why the browser could not join the call's room", () => {
    expect(roomJoinFailureOf(new RoomJoinTimeout())).toBe("timeout");
    // LiveKit refusing the join: a plan past its allowance, or a token it
    // would not take.
    expect(
      roomJoinFailureOf({ name: "ConnectionError", reasonName: "NotAllowed" }),
    ).toBe("refused");
    expect(
      roomJoinFailureOf({
        name: "ConnectionError",
        reasonName: "ServerUnreachable",
        status: 429,
      }),
    ).toBe("refused");
    expect(
      roomJoinFailureOf({
        name: "ConnectionError",
        reasonName: "ServerUnreachable",
        status: 503,
      }),
    ).toBe("failed");
    expect(roomJoinFailureOf(new Error("the SDK could not load"))).toBe(
      "failed",
    );
    expect(roomJoinFailureOf(null)).toBe("failed");
  });
});
