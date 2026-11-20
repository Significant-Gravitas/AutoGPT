import type { FinalizeResponseOttoReplyKind } from "@/app/api/__generated__/models/finalizeResponseOttoReplyKind";

// What Otto says on the not-enough screen, and after it. He answers what he
// heard the way a person would, then asks one big, easy question.

export const OFF_YOUR_PLATE_QUESTION = "What do you want off your plate?";
export const FEW_SENTENCES_HINT = "A few sentences is plenty.";
export const NO_WORDS_HEARD = "No words came through";

export type InsufficientMode = "voice" | "typed";

// `heard`: a voice take with words in it. `silent`: a voice take with none
// Otto can show. `typed`: text that was too short. These are the defaults,
// for when no line below was picked for what was said.
type ReplyCase = "heard" | "silent" | "typed";

const FIRST_LINES: Record<ReplyCase, string> = {
  heard: "Got it. So tell me,",
  silent: "I couldn't hear any words. Check that your mic is on, then tell me,",
  typed: "No problem. Let's start with an easy one.",
};

export type OttoReplyKind = NonNullable<FinalizeResponseOttoReplyKind>;

// One line per kind of too-short take, picked by the backend
// (`otto_reply.py`, with Jev) for what was said. Each leads into the big
// question under it, and each is checked to fit the bubble on a 320px
// phone. "other" means none of these fit, so it gets the default.
export const REPLY_LINES: Record<Exclude<OttoReplyKind, "other">, string> = {
  mic_check: "Loud and clear! So tell me,",
  greeting: "Hi, great to meet you! So tell me,",
  unsure: "No pressure, any answer works. So tell me,",
  reluctant: "No worries, we'll keep it quick. So tell me,",
  what_to_say: "Just one task you'd hand off is plenty.",
  wants_to_skip: "You can skip below. But first, tell me,",
  about_otto: "I'm Otto, here to take work off your hands.",
  busy: "No rush, take your time. When you're ready,",
  annoyed: "Sorry, I'll keep this quick. So tell me,",
  some_work: "Thanks, that's a good start! So tell me,",
  garbled: "I only caught a little of that. So tell me,",
  test: "Got your test! So tell me,",
};

const ACTION_LABELS: Record<InsufficientMode, { talk: string; write: string }> =
  {
    voice: { talk: "Talk", write: "Write" },
    typed: { talk: "Talk instead", write: "Write it" },
  };

export function heardTextOf(text: string | null | undefined) {
  return text?.trim() || null;
}

// Otto answers what was said with the line picked for it; the defaults
// cover no pick, "other", and a take with no words.
export function ottoFirstLine(
  mode: InsufficientMode,
  heardText: string | null,
  replyKind: OttoReplyKind | null = null,
) {
  const heard = heardTextOf(heardText);
  if (heard && replyKind && replyKind !== "other")
    return REPLY_LINES[replyKind];
  if (mode === "typed") return FIRST_LINES.typed;
  return heard ? FIRST_LINES.heard : FIRST_LINES.silent;
}

export function actionLabels(mode: InsufficientMode) {
  return ACTION_LABELS[mode];
}

export function followUpPlaceholder(example: string) {
  return `Like: ${example}`;
}
