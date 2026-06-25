import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { ToolUIPart } from "ai";

interface ClarifyingQuestionPayload {
  question: string;
  keyword: string;
  example?: string;
}

export interface AskQuestionOutput {
  type: string;
  message: string;
  questions: ClarifyingQuestionPayload[];
  session_id?: string;
}

interface ErrorOutput {
  type: "error";
  message: string;
  error?: string;
}

export type AskQuestionToolOutput = AskQuestionOutput | ErrorOutput;

function parseOutput(output: unknown): AskQuestionToolOutput | null {
  if (!output) return null;
  if (typeof output === "string") {
    try {
      return parseOutput(JSON.parse(output) as unknown);
    } catch {
      return null;
    }
  }
  if (typeof output === "object" && output !== null) {
    const obj = output as Record<string, unknown>;
    if (
      obj.type === ResponseType.agent_builder_clarification_needed ||
      "questions" in obj
    ) {
      return obj as unknown as AskQuestionOutput;
    }
    if (obj.type === "error" || "error" in obj) {
      return obj as unknown as ErrorOutput;
    }
  }
  return null;
}

export function getAskQuestionOutput(
  part: ToolUIPart,
): AskQuestionToolOutput | null {
  return parseOutput(part.output);
}

export function isClarificationOutput(
  output: AskQuestionToolOutput,
): output is AskQuestionOutput {
  return (
    output.type === ResponseType.agent_builder_clarification_needed ||
    "questions" in output
  );
}

export function isErrorOutput(
  output: AskQuestionToolOutput,
): output is ErrorOutput {
  return output.type === "error" || "error" in output;
}

export function getAnimationText(part: ToolUIPart): string {
  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return "Asking question...";
    case "output-available": {
      const output = parseOutput(part.output);
      if (output && isClarificationOutput(output)) return "Needs your input";
      if (output && isErrorOutput(output)) return "Failed to ask question";
      return "Asking question...";
    }
    case "output-error":
      return "Failed to ask question";
    default:
      return "Asking question...";
  }
}
