export interface ClarifyingQuestion {
  question: string;
  keyword: string;
  example?: string;
}

export function normalizeClarifyingQuestions(
  questions: Array<{ question: string; keyword: string; example?: unknown }>,
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

export function extractClarifyingQuestions(source: {
  input?: unknown;
  output?: unknown;
}): ClarifyingQuestion[] {
  const raw = questionItems(source.output) ?? questionItems(source.input) ?? [];
  const valid = raw.flatMap((item) =>
    typeof item.question === "string" &&
    item.question.trim() &&
    typeof item.keyword === "string"
      ? [
          {
            question: item.question.trim(),
            keyword: item.keyword,
            example: item.example,
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
