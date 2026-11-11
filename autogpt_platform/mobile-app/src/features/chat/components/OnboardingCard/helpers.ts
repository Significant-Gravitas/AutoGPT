export const ONBOARDING_TOOL = "expert_onboarding";

// The backend matches both strings exactly
// (backend/api/features/experts/onboarding.py), so keep them byte for byte.
export const SKIP_MESSAGE = "Let's skip the setup questions for now.";
const ANSWERS_HEADER = "**Here are my answers:**\n\n";
const ANSWERS_FOOTER = "\n\nPlease proceed.";

export interface OnboardingStep {
  question: string;
  keyword: string;
  options: string[];
  allowMultiple: boolean;
}

/** Tapped options plus the "Other" text. A single-choice step holds one or the other. */
export interface StepAnswer {
  picks: string[];
  custom: string;
}

export interface SentAnswer {
  keyword: string;
  question: string;
  answer: string;
}

export const EMPTY_ANSWER: StepAnswer = { picks: [], custom: "" };

function toRecord(value: unknown): Record<string, unknown> | null {
  if (typeof value === "string") {
    try {
      value = JSON.parse(value) as unknown;
    } catch {
      return null;
    }
  }
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function toStep(value: unknown, index: number): OnboardingStep | null {
  const record = toRecord(value);
  const question = record?.question;
  if (typeof question !== "string" || !question.trim()) return null;
  const keyword =
    typeof record?.keyword === "string" && record.keyword.trim()
      ? record.keyword.trim()
      : `step-${index}`;
  const options = Array.isArray(record?.options)
    ? record.options.flatMap((option) =>
        typeof option === "string" && option.trim() ? [option.trim()] : [],
      )
    : [];
  return {
    question: question.trim(),
    keyword,
    options: Array.from(new Set(options)),
    allowMultiple: options.length > 0 && record?.allow_multiple === true,
  };
}

/** The card's questions from the tool output, or null if it isn't a usable card. */
export function parseOnboarding(output: unknown): OnboardingStep[] | null {
  const record = toRecord(output);
  if (!record || record.type !== ONBOARDING_TOOL) return null;
  if (!Array.isArray(record.steps)) return null;
  const steps = record.steps.flatMap((step, index) => toStep(step, index) ?? []);
  return steps.length > 0 ? steps : null;
}

export function optionLetter(index: number) {
  return String.fromCharCode(65 + index);
}

export function answerList(answer: StepAnswer | undefined) {
  if (!answer) return [];
  return [...answer.picks, answer.custom].flatMap((pick) =>
    pick.trim() ? [pick.trim()] : [],
  );
}

export function isAnswered(answer: StepAnswer | undefined) {
  return answerList(answer).length > 0;
}

/** Several picks become a bullet list, one stays inline, as on the web. */
export function formatAnswer(answer: StepAnswer | undefined) {
  const picks = answerList(answer);
  return picks.length > 1
    ? picks.map((pick) => `- ${pick}`).join("\n")
    : (picks[0] ?? "");
}

export function pickOption(
  step: OnboardingStep,
  answer: StepAnswer | undefined,
  option: string,
): StepAnswer {
  if (!step.allowMultiple) return { picks: [option], custom: "" };
  const current = answer ?? EMPTY_ANSWER;
  const next = current.picks.includes(option)
    ? current.picks.filter((pick) => pick !== option)
    : [...current.picks, option];
  return {
    picks: step.options.filter((pick) => next.includes(pick)),
    custom: current.custom,
  };
}

export function setCustomText(
  step: OnboardingStep,
  answer: StepAnswer | undefined,
  text: string,
): StepAnswer {
  if (!step.allowMultiple) return { picks: [], custom: text };
  return { picks: answer?.picks ?? [], custom: text };
}

export function buildAnswersMessage(
  steps: OnboardingStep[],
  answers: Record<string, StepAnswer>,
) {
  const body = steps
    .flatMap((step) => {
      const answer = formatAnswer(answers[step.keyword]);
      return answer ? [`> ${step.question}\n\n${answer}`] : [];
    })
    .join("\n\n");
  return `${ANSWERS_HEADER}${body}${ANSWERS_FOOTER}`;
}

function displayAnswer(step: OnboardingStep, answer: string) {
  const lines = answer.split("\n");
  return step.allowMultiple && lines.every((line) => line.startsWith("- "))
    ? lines.map((line) => line.slice(2)).join(", ")
    : answer;
}

/** The answers a reply carries for these steps. Empty for a skip or an unrelated reply. */
export function parseSentAnswers(
  text: string,
  steps: OnboardingStep[],
): SentAnswer[] {
  const start = text.indexOf(ANSWERS_HEADER);
  if (start < 0) return [];
  const rest = text.slice(start + ANSWERS_HEADER.length);
  const end = rest.lastIndexOf("\n\nPlease proceed");
  if (end < 0) return [];
  const body = `\n\n${rest.slice(0, end)}`;
  const markers = steps
    .flatMap((step) => {
      const marker = `\n\n> ${step.question}\n\n`;
      const at = body.indexOf(marker);
      return at < 0 ? [] : [{ step, at, answerAt: at + marker.length }];
    })
    .sort((a, b) => a.at - b.at);
  return markers.flatMap((marker, index) => {
    const answer = body.slice(marker.answerAt, markers[index + 1]?.at).trim();
    if (!answer) return [];
    return [
      {
        keyword: marker.step.keyword,
        question: marker.step.question,
        answer: displayAnswer(marker.step, answer),
      },
    ];
  });
}
