export interface ClarifyingQuestion {
  question: string;
  keyword: string;
  example?: string;
  options?: string[];
  /** Several of `options` may be picked. Never set without options. */
  allow_multiple?: boolean;
}

/** A multi-select answer: the ticked options, and any typed text kept in its
 *  own slot rather than mixed in with them. Membership of `options` cannot
 *  tell the two apart — "Research" on the way to "Research and development"
 *  is typing, not a tick — so the shape has to. */
export interface MultiAnswer {
  selected: string[];
  custom: string;
}

/** One question's answer: a single-select pick or typed text is a string, a
 *  multi-select one is a MultiAnswer. */
export type QuestionAnswer = string | MultiAnswer;

/** An answer as its list of picks, blanks dropped — so an empty list is the
 *  one definition of "not answered yet" every caller shares. */
export function toAnswerList(answer: QuestionAnswer | undefined): string[] {
  const picks =
    typeof answer === "string" || answer === undefined
      ? [answer ?? ""]
      : [...answer.selected, answer.custom];
  return picks.flatMap((pick) => (pick.trim() ? [pick.trim()] : []));
}

export function isAnswered(answer: QuestionAnswer | undefined): boolean {
  return toAnswerList(answer).length > 0;
}

/** An answer as the raw text of a single-answer field. Untrimmed, so the
 *  space the user just typed survives the round trip through state. */
export function toAnswerText(answer: QuestionAnswer | undefined): string {
  return typeof answer === "string" ? answer : "";
}

/** An answer as a multi-select field reads it. A string answer was never a
 *  pick, so it becomes the typed text. */
export function toMultiAnswer(answer: QuestionAnswer | undefined): MultiAnswer {
  if (typeof answer === "string") return { selected: [], custom: answer };
  return answer ?? { selected: [], custom: "" };
}

/** The answer as it reads in the message sent back: several picks become a
 *  bullet list, one stays inline so single-select replies are unchanged. */
export function formatAnswer(answer: QuestionAnswer | undefined): string {
  const picks = toAnswerList(answer);
  return picks.length > 1
    ? picks.map((pick) => `- ${pick}`).join("\n")
    : (picks[0] ?? "");
}

function toOptions(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const options = value.flatMap((option) =>
    typeof option === "string" && option.trim() ? [option.trim()] : [],
  );
  return options.length > 0 ? Array.from(new Set(options)) : undefined;
}

export function normalizeClarifyingQuestions(
  questions: Array<{
    question: string;
    keyword: string;
    example?: unknown;
    options?: unknown;
    allow_multiple?: unknown;
  }>,
): ClarifyingQuestion[] {
  const seen = new Set<string>();

  return questions.map((q, index) => {
    let keyword = q.keyword?.trim().toLowerCase() || "";
    if (!keyword) {
      keyword = `question-${index}`;
    }

    let unique = keyword;
    let suffix = 1;
    while (seen.has(unique)) {
      unique = `${keyword}-${suffix}`;
      suffix++;
    }
    seen.add(unique);

    const item: ClarifyingQuestion = {
      question: q.question,
      keyword: unique,
    };
    const example =
      typeof q.example === "string" && q.example.trim()
        ? q.example.trim()
        : null;
    if (example) item.example = example;
    const options = toOptions(q.options);
    if (options) item.options = options;
    // Without options the card has nothing to toggle, so the flag would only
    // promise a multi-select the user never gets.
    if (options && q.allow_multiple === true) item.allow_multiple = true;
    return item;
  });
}

function toRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return null;
    }
  }
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function questionItems(value: unknown): Record<string, unknown>[] | null {
  const raw = toRecord(value)?.questions;
  if (!Array.isArray(raw) || raw.length === 0) return null;
  const items = raw.filter(
    (item): item is Record<string, unknown> =>
      !!item && typeof item === "object" && !Array.isArray(item),
  );
  return items.length > 0 ? items : null;
}

/** Identifies a question across the tool call's input and output. Question text
 *  alone is not unique — the same prompt can be asked for two keywords (e.g. a
 *  source and a destination channel) with different options each.
 *
 *  The separator must be a character that cannot appear in either half, hence
 *  an escaped NUL: with a space, keyword "a" + question "b c" and keyword "a b" +
 *  question "c" would collide. Keep it escaped — a literal NUL byte makes the
 *  whole file binary to git and unreviewable on GitHub. */
function recoveryKey(item: Record<string, unknown>): string {
  const keyword =
    typeof item.keyword === "string" ? item.keyword.trim().toLowerCase() : "";
  const question =
    typeof item.question === "string" ? item.question.trim() : "";
  return `${keyword}\u0000${question}`;
}

export function extractClarifyingQuestions(source: {
  input?: unknown;
  output?: unknown;
}): ClarifyingQuestion[] {
  const fromInput = questionItems(source.input);
  const raw = questionItems(source.output) ?? fromInput ?? [];
  // Older tool outputs collapse options into the example string and predate
  // allow_multiple, so when the output is the source both only survive in the
  // input the model actually sent.
  const fromInputByKey = new Map<string, Record<string, unknown>>();
  for (const item of fromInput ?? []) {
    fromInputByKey.set(recoveryKey(item), item);
  }
  const valid = raw.flatMap((item) => {
    if (
      typeof item.question !== "string" ||
      !item.question.trim() ||
      typeof item.keyword !== "string"
    ) {
      return [];
    }
    const asked = fromInputByKey.get(recoveryKey(item));
    return [
      {
        question: item.question.trim(),
        keyword: item.keyword,
        example: item.example,
        options: toOptions(item.options) ?? toOptions(asked?.options),
        allow_multiple:
          item.allow_multiple === true || asked?.allow_multiple === true,
      },
    ];
  });
  return normalizeClarifyingQuestions(valid);
}

/**
 * Formats clarification answers as a context message and sends it via onSend.
 */
export function buildClarificationAnswersMessage(
  answers: Record<string, QuestionAnswer>,
  rawQuestions: Array<{ question: string; keyword: string }>,
  mode: "create" | "edit",
): string {
  const contextMessage = rawQuestions
    .map((q) => `> ${q.question}\n\n${formatAnswer(answers[q.keyword])}`)
    .join("\n\n");

  const action = mode === "create" ? "creating" : "editing";
  return `**Here are my answers:**\n\n${contextMessage}\n\nPlease proceed with ${action} the agent.`;
}
