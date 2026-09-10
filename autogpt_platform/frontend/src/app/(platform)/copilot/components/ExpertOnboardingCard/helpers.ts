import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import type { ClarifyingQuestion } from "../../tools/clarifying-questions";

export const EXPERT_ONBOARDING_PART_TYPE = "tool-expert_onboarding";

export interface ExpertOnboardingStep {
  question: string;
  keyword: string;
  options: string[];
}

export interface ExpertOnboardingOutput {
  expertId: string | null;
  greeting: string;
  steps: ExpertOnboardingStep[];
}

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

function toStep(value: unknown, index: number): ExpertOnboardingStep | null {
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
  return { question: question.trim(), keyword, options };
}

export function parseExpertOnboarding(
  part: ToolUIPart,
): ExpertOnboardingOutput | null {
  const output = toRecord(part.output);
  if (!output || output.type !== ResponseType.expert_onboarding) return null;
  const steps = Array.isArray(output.steps)
    ? output.steps.flatMap((step, index) => {
        const parsed = toStep(step, index);
        return parsed ? [parsed] : [];
      })
    : [];
  if (steps.length === 0) return null;
  return {
    expertId: typeof output.expert_id === "string" ? output.expert_id : null,
    greeting: typeof output.greeting === "string" ? output.greeting : "",
    steps,
  };
}

/** The card reuses the tool chain's answer field, which speaks
 *  `ClarifyingQuestion`. `example` stays empty: the field only shows it when
 *  a question has no options, and an onboarding step without options is an
 *  open "anything else?" that reads better with the plain placeholder. */
export function toClarifyingQuestion(
  step: ExpertOnboardingStep,
): ClarifyingQuestion {
  return step.options.length > 0
    ? { question: step.question, keyword: step.keyword, options: step.options }
    : { question: step.question, keyword: step.keyword };
}

export function buildOnboardingAnswersMessage(
  steps: ExpertOnboardingStep[],
  answers: Record<string, string>,
): string {
  const body = steps
    .flatMap((step) => {
      const answer = (answers[step.keyword] ?? "").trim();
      return answer ? [`> ${step.question}\n\n${answer}`] : [];
    })
    .join("\n\n");
  return `**Here are my answers:**\n\n${body}\n\nPlease proceed.`;
}

type Message = UIMessage<unknown, UIDataTypes, UITools>;

/** Tool-call id of the onboarding card still waiting on the user.
 *
 * Same rule as the clarifying-question dock: a card is live only while the
 * assistant message that opened it is the last one. Any user reply — the
 * answers, a skip, or an unrelated request — settles it, so a reloaded
 * thread renders history rather than a form that would send the answers
 * twice. */
export function getPendingOnboardingCallId(messages: Message[]): string | null {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant") return null;
  for (const part of last.parts) {
    if (part.type !== EXPERT_ONBOARDING_PART_TYPE) continue;
    const tool = part as ToolUIPart;
    if (tool.state !== "output-available") continue;
    if (parseExpertOnboarding(tool)) return tool.toolCallId;
  }
  return null;
}
