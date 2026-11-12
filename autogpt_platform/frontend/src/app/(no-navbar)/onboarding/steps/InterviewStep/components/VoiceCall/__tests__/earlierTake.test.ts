import type { VoiceCallStatusResponse } from "@/app/api/__generated__/models/voiceCallStatusResponse";
import { describe, expect, it } from "vitest";
import { endingWithEarlierTake, savedBeforeCopy } from "../earlierTake";

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

describe("a call that left no take, beside the take before it", () => {
  const EARLIER_ID = "e".repeat(32);
  // The server's read of the person's latest call, looking back past calls
  // without words: here an earlier call they hung up, its words saved.
  const hungUpBefore = call({ call_id: EARLIER_ID, end_reason: "user_hangup" });

  it.each([
    ["no_words", { kind: "no_words" }],
    ["unsaved", { kind: "unsaved", reason: "user_hangup" }],
  ] as const)(
    "offers the earlier call's saved words after a call that left %s",
    (latest, ending) => {
      expect(endingWithEarlierTake(ending, hungUpBefore)).toEqual({
        kind: "saved_before",
        latest,
      });
    },
  );

  it("offers an earlier call's words the pipeline never ruled on the same way", () => {
    const neverRuled = call({
      call_id: EARLIER_ID,
      status: "failed",
      end_reason: "vendor_error",
      hand_off: failed("quality_check_unavailable"),
    });
    expect(endingWithEarlierTake({ kind: "no_words" }, neverRuled)).toEqual({
      kind: "saved_before",
      latest: "no_words",
    });
  });

  it("shows an earlier call that took the interview as it stands", () => {
    const took = call({ call_id: EARLIER_ID, interview: "taken" });
    expect(endingWithEarlierTake({ kind: "no_words" }, took)).toEqual({
      kind: "team",
      path: "A",
    });
  });

  it.each([
    ["no earlier call", null],
    [
      "an earlier call with too few words",
      call({
        call_id: EARLIER_ID,
        status: "failed",
        hand_off: failed("insufficient_content"),
      }),
    ],
    [
      "an earlier call the person's typing superseded",
      call({ call_id: EARLIER_ID, interview: "superseded" }),
    ],
    [
      "an earlier call still on its way to a take",
      call({ call_id: EARLIER_ID, status: "transcribing", hand_off: null }),
    ],
  ])("keeps the latest call's own ending with %s", (_case, earlier) => {
    expect(endingWithEarlierTake({ kind: "no_words" }, earlier)).toEqual({
      kind: "no_words",
    });
  });

  it.each([
    [{ kind: "insufficient" }],
    [{ kind: "lost", reason: "client_gone" }],
    [{ kind: "retry", leftOpen: true, reason: "client_gone" }],
  ] as const)(
    "keeps the ending of a call whose words are its own take (%o)",
    (ending) => {
      expect(endingWithEarlierTake(ending, hungUpBefore)).toEqual(ending);
    },
  );

  it("says what became of the latest call's words, and that the earlier ones stand", () => {
    expect(savedBeforeCopy("no_words")).toBe(
      "We didn't hear you on that call. What you said before is still saved.",
    );
    expect(savedBeforeCopy("unsaved")).toBe(
      "Something went wrong saving that call. What you said before is still saved.",
    );
  });
});
