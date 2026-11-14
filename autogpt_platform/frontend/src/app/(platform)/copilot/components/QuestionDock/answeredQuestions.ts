import type { ToolUIPart, UIDataTypes, UIMessage, UITools } from "ai";
import {
  type ClarifyingQuestion,
  extractClarifyingQuestions,
} from "../../tools/clarifying-questions";
import {
  EXPERT_ONBOARDING_PART_TYPE,
  parseExpertOnboarding,
} from "../ExpertOnboardingCard/helpers";
import { isQuestionTool } from "./helpers";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

export type AnsweredQuestions = ReadonlyMap<
  string,
  ReadonlyMap<string, string>
>;

interface ParsedAnswer {
  question: string;
  answer: string;
}

function parseAnswersMessage(
  text: string,
  questions: string[],
): ParsedAnswer[] {
  const answers: ParsedAnswer[] = [];
  const prompts = questions.toSorted((a, b) => b.length - a.length);
  for (const section of text.split("**Here are my answers:**\n\n").slice(1)) {
    const footer = section.lastIndexOf("\n\nPlease proceed");
    if (footer < 0) continue;
    const body = section.slice(0, footer);
    const markers = [...body.matchAll(/(?:^|\n\n)> /g)].flatMap((match) => {
      const start = match.index + match[0].length;
      const question = prompts.find((prompt) =>
        body.startsWith(`${prompt}\n\n`, start),
      );
      return question
        ? [
            {
              question,
              start: match.index,
              answerStart: start + question.length + 2,
            },
          ]
        : [];
    });
    for (const [index, marker] of markers.entries()) {
      answers.push({
        question: marker.question,
        answer: body
          .slice(marker.answerStart, markers[index + 1]?.start)
          .trim(),
      });
    }
  }
  return answers;
}

function askedQuestions(part: ToolUIPart): ClarifyingQuestion[] {
  if (part.state !== "output-available") return [];
  if (part.type === EXPERT_ONBOARDING_PART_TYPE) {
    return parseExpertOnboarding(part)?.steps ?? [];
  }
  return isQuestionTool(part.type.slice("tool-".length), part.output)
    ? extractClarifyingQuestions(part)
    : [];
}

function displayAnswer(question: ClarifyingQuestion, answer: string): string {
  const lines = answer.split("\n");
  return question.allow_multiple && lines.every((line) => line.startsWith("- "))
    ? lines.map((line) => line.slice(2)).join(", ")
    : answer;
}

function answersForMessage(message: Message, reply: Message) {
  const text = reply.parts
    .flatMap((part) => (part.type === "text" ? [part.text] : []))
    .join("\n");
  const questionsByCall = message.parts.flatMap((part) => {
    if (!part.type.startsWith("tool-")) return [];
    const tool = part as ToolUIPart;
    return [{ callId: tool.toolCallId, questions: askedQuestions(tool) }];
  });
  const remaining = parseAnswersMessage(
    text,
    questionsByCall.flatMap((call) => call.questions.map((q) => q.question)),
  );
  const calls = new Map<string, ReadonlyMap<string, string>>();
  for (const { callId, questions } of questionsByCall) {
    const answers = new Map<string, string>();
    for (const question of questions) {
      const index = remaining.findIndex(
        (entry) => entry.question === question.question,
      );
      if (index < 0) continue;
      const [entry] = remaining.splice(index, 1);
      if (entry.answer) {
        answers.set(question.keyword, displayAnswer(question, entry.answer));
      }
    }
    if (answers.size > 0) calls.set(callId, answers);
  }
  return calls;
}

export function getAnsweredQuestions(messages: Message[]): AnsweredQuestions {
  const answered = new Map<string, ReadonlyMap<string, string>>();
  let nextReply: Message | null = null;
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index];
    if (message.role === "user") {
      nextReply = message;
    } else if (message.role === "assistant" && nextReply) {
      for (const [callId, answers] of answersForMessage(message, nextReply)) {
        answered.set(callId, answers);
      }
    }
  }
  return answered;
}

const ANSWERS_HEADER = "**Here are my answers:**\n\n";
const ANSWERS_FOOTER = "\n\nPlease proceed";

export interface AnswerLine {
  question: string;
  answer: string;
}

/** The questions and answers of a message the question card sent, or null
 *  for any other message. The frame stays in the stored text (the backend
 *  reads it); the thread shows it as "Answered: ..." over the answer. */
export function parseAnswersFrame(text: string): AnswerLine[] | null {
  const trimmed = text.trim();
  if (!trimmed.startsWith(ANSWERS_HEADER.trim())) return null;
  const footer = trimmed.lastIndexOf(ANSWERS_FOOTER);
  if (footer < 0) return null;
  const body = trimmed.slice(ANSWERS_HEADER.trim().length, footer).trim();
  const lines = body
    .split(/(?:^|\n\n)> /)
    .filter((block) => block.trim())
    .flatMap((block) => {
      const split = block.indexOf("\n\n");
      if (split < 0) return [];
      return [
        {
          question: block.slice(0, split).trim(),
          answer: block.slice(split + 2).trim(),
        },
      ];
    });
  return lines.length > 0 ? lines : null;
}
