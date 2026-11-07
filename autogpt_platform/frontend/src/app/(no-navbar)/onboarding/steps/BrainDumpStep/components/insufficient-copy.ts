// What Otto says on the not-enough screen, and after it. He answers what he
// heard the way a person would, then asks one big, easy question.

export const OFF_YOUR_PLATE_QUESTION = "What do you want off your plate?";
export const FEW_SENTENCES_HINT = "A few sentences is plenty.";
export const NO_WORDS_HEARD = "No words came through";

export type InsufficientMode = "voice" | "typed";

// `heard`: a voice take with words in it. `silent`: a voice take with none
// Otto can show. `typed`: text that was too short.
type ReplyCase = "heard" | "silent" | "typed";

const FIRST_LINES: Record<ReplyCase, string> = {
  heard: "Loud and clear! So tell me,",
  silent: "I couldn't hear any words. Check that your mic is on, then tell me,",
  typed: "No problem. Let's start with an easy one.",
};

const ACTION_LABELS: Record<InsufficientMode, { talk: string; write: string }> =
  {
    voice: { talk: "Talk", write: "Write" },
    typed: { talk: "Talk instead", write: "Write it" },
  };

export function heardTextOf(text: string | null | undefined) {
  return text?.trim() || null;
}

// Otto answers what was said when the gate's judge wrote him a reply; the
// fixed lines cover a reply that is missing and a take with no words.
export function ottoFirstLine(
  mode: InsufficientMode,
  heardText: string | null,
  ottoReply: string | null = null,
) {
  const heard = heardTextOf(heardText);
  const written = heardTextOf(ottoReply);
  if (heard && written) return written;
  if (mode === "typed") return FIRST_LINES.typed;
  return heard ? FIRST_LINES.heard : FIRST_LINES.silent;
}

export function actionLabels(mode: InsufficientMode) {
  return ACTION_LABELS[mode];
}

export function followUpPlaceholder(example: string) {
  return `Like: ${example}`;
}
