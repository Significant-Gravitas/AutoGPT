export interface ClarifyingQuestion {
  question: string;
  keyword: string;
  example?: string;
  options?: string[];
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
  // Older tool outputs collapse options into the example string, so when the
  // output is the source the selectable options only survive in the input.
  // This runs per render on an unmemoized chain row, so skip the whole map
  // when every item already carries its own options.
  const inputOptions = new Map<string, string[]>();
  if (raw.some((item) => !toOptions(item.options))) {
    for (const item of fromInput ?? []) {
      const options = toOptions(item.options);
      if (options) inputOptions.set(recoveryKey(item), options);
    }
  }
  const valid = raw.flatMap((item) =>
    typeof item.question === "string" &&
    item.question.trim() &&
    typeof item.keyword === "string"
      ? [
          {
            question: item.question.trim(),
            keyword: item.keyword,
            example: item.example,
            options:
              toOptions(item.options) ?? inputOptions.get(recoveryKey(item)),
          },
        ]
      : [],
  );
  return normalizeClarifyingQuestions(valid);
}

/**
 * Formats clarification answers as a context message and sends it via onSend.
 */
export function buildClarificationAnswersMessage(
  answers: Record<string, string>,
  rawQuestions: Array<{ question: string; keyword: string }>,
  mode: "create" | "edit",
): string {
  const contextMessage = rawQuestions
    .map((q) => {
      const answer = answers[q.keyword] || "";
      return `> ${q.question}\n\n${answer}`;
    })
    .join("\n\n");

  const action = mode === "create" ? "creating" : "editing";
  return `**Here are my answers:**\n\n${contextMessage}\n\nPlease proceed with ${action} the agent.`;
}
