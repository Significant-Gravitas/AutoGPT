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
