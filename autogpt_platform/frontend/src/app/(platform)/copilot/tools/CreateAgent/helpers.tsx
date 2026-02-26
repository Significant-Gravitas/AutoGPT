import type { AgentPreviewResponse } from "@/app/api/__generated__/models/agentPreviewResponse";
import type { AgentSavedResponse } from "@/app/api/__generated__/models/agentSavedResponse";
import type { ClarificationNeededResponse } from "@/app/api/__generated__/models/clarificationNeededResponse";
import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { SuggestedGoalResponse } from "@/app/api/__generated__/models/suggestedGoalResponse";
import {
  PlusCircleIcon,
  PlusIcon,
  WarningDiamondIcon,
} from "@phosphor-icons/react";
import type { ToolUIPart } from "ai";
import { OrbitLoader } from "../../components/OrbitLoader/OrbitLoader";

export type CreateAgentToolOutput =
  | AgentPreviewResponse
  | AgentSavedResponse
  | ClarificationNeededResponse
  | SuggestedGoalResponse
  | ErrorResponse;

function parseOutput(output: unknown): CreateAgentToolOutput | null {
  if (!output) return null;
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (!trimmed) return null;
    try {
      return parseOutput(JSON.parse(trimmed) as unknown);
    } catch {
      return null;
    }
  }
  if (typeof output === "object") {
    const type = (output as { type?: unknown }).type;
    if (
      type === ResponseType.agent_preview ||
      type === ResponseType.agent_saved ||
      type === ResponseType.clarification_needed ||
      type === ResponseType.suggested_goal ||
      type === ResponseType.error
    ) {
      return output as CreateAgentToolOutput;
    }
    if ("agent_json" in output && "agent_name" in output)
      return output as AgentPreviewResponse;
    if ("agent_id" in output && "library_agent_id" in output)
      return output as AgentSavedResponse;
    if ("questions" in output) return output as ClarificationNeededResponse;
    if ("suggested_goal" in output) return output as SuggestedGoalResponse;
    if ("error" in output || "details" in output)
      return output as ErrorResponse;
  }
  return null;
}

export function getCreateAgentToolOutput(
  part: unknown,
): CreateAgentToolOutput | null {
  if (!part || typeof part !== "object") return null;
  return parseOutput((part as { output?: unknown }).output);
}

export function isAgentPreviewOutput(
  output: CreateAgentToolOutput,
): output is AgentPreviewResponse {
  return output.type === ResponseType.agent_preview || "agent_json" in output;
}

export function isAgentSavedOutput(
  output: CreateAgentToolOutput,
): output is AgentSavedResponse {
  return (
    output.type === ResponseType.agent_saved || "agent_page_link" in output
  );
}

export function isClarificationNeededOutput(
  output: CreateAgentToolOutput,
): output is ClarificationNeededResponse {
  return (
    output.type === ResponseType.clarification_needed || "questions" in output
  );
}

export function isSuggestedGoalOutput(
  output: CreateAgentToolOutput,
): output is SuggestedGoalResponse {
  return (
    output.type === ResponseType.suggested_goal || "suggested_goal" in output
  );
}

export function isErrorOutput(
  output: CreateAgentToolOutput,
): output is ErrorResponse {
  return output.type === ResponseType.error || "error" in output;
}

export function getAnimationText(part: {
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}): string {
  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return "Creating a new agent";
    case "output-available": {
      const output = parseOutput(part.output);
      if (!output) return "Creating a new agent";
      if (isAgentSavedOutput(output)) return `Saved ${output.agent_name}`;
      if (isAgentPreviewOutput(output)) return `Preview "${output.agent_name}"`;
      if (isClarificationNeededOutput(output)) return "Needs clarification";
      if (isSuggestedGoalOutput(output)) return "Goal needs refinement";
      return "Error creating agent";
    }
    case "output-error":
      return "Error creating agent";
    default:
      return "Creating a new agent";
  }
}

export function ToolIcon({
  isStreaming,
  isError,
}: {
  isStreaming?: boolean;
  isError?: boolean;
}) {
  if (isError) {
    return (
      <WarningDiamondIcon size={14} weight="regular" className="text-red-500" />
    );
  }
  if (isStreaming) {
    return <OrbitLoader size={24} />;
  }
  return <PlusIcon size={14} weight="regular" className="text-neutral-400" />;
}

export function AccordionIcon() {
  return <PlusCircleIcon size={32} weight="light" />;
}

export function formatMaybeJson(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function truncateText(text: string, maxChars: number): string {
  const trimmed = text.trim();
  if (trimmed.length <= maxChars) return trimmed;
  return `${trimmed.slice(0, maxChars).trimEnd()}…`;
}

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
