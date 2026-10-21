import type { ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import type { Dispatch, SetStateAction } from "react";
import {
  type ClarifyingQuestion,
  type QuestionAnswer,
  extractClarifyingQuestions,
  formatAnswer,
  isAnswered,
  normalizeClarifyingQuestions,
} from "../../tools/clarifying-questions";
import { EXPERT_ONBOARDING_PART_TYPE } from "../ExpertOnboardingCard/helpers";
import type { ChainActionEntry } from "../ToolChain/chainActions";

export interface PendingQuestions {
  dockId: string;
  questions: ClarifyingQuestion[];
  /** Tool-call ids of the ask_question parts the questions came from —
   *  lets the tool chain render the answer form on the matching rows. */
  callIds: string[];
}

type Message = UIMessage<unknown, UIDataTypes, UITools>;

const QUESTION_RESPONSE_TYPE = "agent_builder_clarification_needed";

function outputType(output: unknown): string | null {
  if (typeof output === "string") {
    try {
      output = JSON.parse(output);
    } catch {
      return null;
    }
  }
  if (!output || typeof output !== "object") return null;
  const type = (output as Record<string, unknown>).type;
  return typeof type === "string" ? type : null;
}

/** Whether a tool call asked the user clarifying questions. `toolName` is the
 *  name without the part's `tool-` prefix. */
export function isQuestionTool(
  toolName: string | undefined,
  output: unknown,
): boolean {
  return (
    toolName === "ask_question" || outputType(output) === QUESTION_RESPONSE_TYPE
  );
}

// Questions are pending only while the asking assistant message is the last
// message — any user reply (answers, skip, or a fresh request) supersedes it.
export function getPendingQuestions(
  messages: Message[],
): PendingQuestions | null {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant") return null;

  const questions: ClarifyingQuestion[] = [];
  const callIds: string[] = [];
  for (const part of last.parts) {
    if (!part.type.startsWith("tool-")) continue;
    if (!("state" in part) || part.state !== "output-available") continue;
    const tool = part as ToolUIPart;
    if (!isQuestionTool(part.type.slice("tool-".length), tool.output)) continue;
    const extracted = extractClarifyingQuestions(tool);
    if (extracted.length === 0) continue;
    questions.push(...extracted);
    callIds.push(tool.toolCallId);
  }
  if (questions.length === 0) return null;

  return {
    dockId: `${last.id}:${callIds.join("+")}`,
    questions: normalizeClarifyingQuestions(questions),
    callIds,
  };
}

const ANSWERS_HEADER = "**Here are my answers:**\n\n";
const ANSWERS_FOOTER = "\n\nPlease proceed";

/** The answers in a reply sent from a question card, by question text. The
 *  reply quotes each question and puts its answer under it; a chain's reply
 *  can carry several cards' sections, each with its own header. */
export function parseAnswersMessage(text: string): Map<string, string> {
  const answers = new Map<string, string>();
  for (const section of text.split(ANSWERS_HEADER).slice(1)) {
    const body = section.split(ANSWERS_FOOTER)[0];
    for (const block of body.split(/\n\n(?=> )/)) {
      const match = block.match(/^> (.+)\n\n([\s\S]*)$/);
      if (!match) continue;
      // Several picks went out as a bullet list; read back, they fit a line.
      const answer = match[2]
        .split("\n")
        .map((line) => line.replace(/^- /, "").trim())
        .filter(Boolean)
        .join(", ");
      if (answer) answers.set(match[1].trim(), answer);
    }
  }
  return answers;
}

/** Answers by question text, keyed by the tool-call id that asked. */
export type AnsweredQuestions = ReadonlyMap<
  string,
  ReadonlyMap<string, string>
>;

function messageText(message: Message): string {
  return message.parts
    .flatMap((part) => (part.type === "text" ? [part.text] : []))
    .join("\n");
}

/** What the user answered to each question card they have moved past, read
 *  from the first user message after the asking one, so a reload shows the
 *  same answers the user sent. */
export function getAnsweredQuestions(messages: Message[]): AnsweredQuestions {
  const answered = new Map<string, ReadonlyMap<string, string>>();
  let nextReply: Message | null = null;
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index];
    if (message.role === "user") {
      nextReply = message;
      continue;
    }
    if (message.role !== "assistant" || !nextReply) continue;
    const callIds = message.parts.flatMap((part) => {
      if (!part.type.startsWith("tool-")) return [];
      const tool = part as ToolUIPart;
      const asks =
        part.type === EXPERT_ONBOARDING_PART_TYPE ||
        isQuestionTool(part.type.slice("tool-".length), tool.output);
      return asks ? [tool.toolCallId] : [];
    });
    if (callIds.length === 0) continue;
    const answers = parseAnswersMessage(messageText(nextReply));
    if (answers.size === 0) continue;
    for (const callId of callIds) answered.set(callId, answers);
  }
  return answered;
}

export function buildAnswersMessage(
  questions: ClarifyingQuestion[],
  answers: Record<string, QuestionAnswer>,
): string {
  const body = questions
    .map((q) => `> ${q.question}\n\n${formatAnswer(answers[q.keyword])}`)
    .join("\n\n");
  return `**Here are my answers:**\n\n${body}\n\nPlease proceed.`;
}

type Answers = Record<string, QuestionAnswer>;

interface QuestionsEntryArgs {
  dockId: string;
  questions: ClarifyingQuestion[];
  answers: Answers;
  register: (entry: ChainActionEntry) => void;
  setAnswers: Dispatch<SetStateAction<Answers>>;
  dismiss: () => void;
}

/** The questions card's entry in the tool chain. An answer re-registers the
 *  entry in the same event as the keystroke: left to the asking component's
 *  effect, the field would get its value one render late, and React writing
 *  the old value back in between moves the caret to the end and cancels an
 *  IME composition (Japanese "nihongo" came out as "nいhおnngお"). */
export function buildQuestionsEntry(
  args: QuestionsEntryArgs,
): ChainActionEntry {
  const { dockId, questions, answers, register, setAnswers, dismiss } = args;
  const ready = questions.every((q) => isAnswered(answers[q.keyword]));
  return {
    id: dockId,
    ready,
    // Answers are ready on the first character of the last one; the user
    // reviews them through the dock's own action, not an automatic send.
    manualProceed: true,
    buildMessage: () =>
      ready ? buildAnswersMessage(questions, answers) : null,
    onSent: dismiss,
    questions: {
      id: dockId,
      questions,
      answers,
      onAnswer: (keyword, value) => {
        // State takes the functional update so a second answer in the same
        // event can't drop the first; the entry registered here is what the
        // field shows this render, and the effect re-registers from state.
        setAnswers((prev) => ({ ...prev, [keyword]: value }));
        register(
          buildQuestionsEntry({
            ...args,
            answers: { ...answers, [keyword]: value },
          }),
        );
      },
      onSkip: dismiss,
    },
  };
}
