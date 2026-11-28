import type { VoiceCallStatusResponse } from "@/app/api/__generated__/models/voiceCallStatusResponse";
import { answerOf, type CallEnding } from "./helpers";

type NoTake = Extract<CallEnding, { kind: "no_words" | "unsaved" }>;

// What became of the latest call's words, beside an earlier call's words
// that still wait for the person.
const SAVED_BEFORE_COPY: Record<NoTake["kind"], string> = {
  no_words:
    "We didn't hear you on that call. What you said before is still saved.",
  unsaved:
    "Something went wrong saving that call. What you said before is still saved.",
};

// A call that left no words to hand off, as nothing was heard or nothing
// was saved, is no take: the take before it, if one stands, is still the
// person's.
export function leavesNoTake(ending: CallEnding): ending is NoTake {
  return ending.kind === "no_words" || ending.kind === "unsaved";
}

// The ending to show for a call that left no take, beside the server's read
// of the person's latest call, which looks back past such calls as a reload
// does (`earlier`: null when it answered this call, or nothing). The words
// of an earlier call still waiting for the person, cut off, hung up or never
// ruled on, are offered after a word on the latest call; an earlier call
// that leads on shows as it stands. Anything else keeps the latest call's
// own ending.
export function endingWithEarlierTake(
  ending: CallEnding,
  earlier: VoiceCallStatusResponse | null,
): CallEnding {
  if (!earlier || !leavesNoTake(ending)) return ending;
  const before = answerOf(earlier);
  if (before === null || before === "typed") return ending;
  if (before.kind === "team") return before;
  if (before.kind === "lost" || before.kind === "retry") {
    return { kind: "saved_before", latest: ending.kind };
  }
  return ending;
}

export function savedBeforeCopy(latest: NoTake["kind"]) {
  return SAVED_BEFORE_COPY[latest];
}
