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
