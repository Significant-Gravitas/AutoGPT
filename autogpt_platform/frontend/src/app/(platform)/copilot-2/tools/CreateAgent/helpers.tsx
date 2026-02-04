import type { ToolUIPart } from "ai";
import {
  CheckCircleIcon,
  CircleNotchIcon,
  XCircleIcon,
} from "@phosphor-icons/react";
import type { AgentPreviewResponse } from "@/app/api/__generated__/models/agentPreviewResponse";
import type { AgentSavedResponse } from "@/app/api/__generated__/models/agentSavedResponse";
import type { ClarificationNeededResponse } from "@/app/api/__generated__/models/clarificationNeededResponse";
import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import type { OperationInProgressResponse } from "@/app/api/__generated__/models/operationInProgressResponse";
import type { OperationPendingResponse } from "@/app/api/__generated__/models/operationPendingResponse";
import type { OperationStartedResponse } from "@/app/api/__generated__/models/operationStartedResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";

export type CreateAgentToolOutput =
  | OperationStartedResponse
  | OperationPendingResponse
  | OperationInProgressResponse
  | AgentPreviewResponse
  | AgentSavedResponse
  | ClarificationNeededResponse
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
      type === ResponseType.operation_started ||
      type === ResponseType.operation_pending ||
      type === ResponseType.operation_in_progress ||
      type === ResponseType.agent_preview ||
      type === ResponseType.agent_saved ||
      type === ResponseType.clarification_needed ||
      type === ResponseType.error
    ) {
      return output as CreateAgentToolOutput;
    }
    if ("operation_id" in output && "tool_name" in output)
      return output as OperationStartedResponse | OperationPendingResponse;
    if ("tool_call_id" in output) return output as OperationInProgressResponse;
    if ("agent_json" in output && "agent_name" in output)
      return output as AgentPreviewResponse;
    if ("agent_id" in output && "library_agent_id" in output)
      return output as AgentSavedResponse;
    if ("questions" in output) return output as ClarificationNeededResponse;
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

export function isOperationStartedOutput(
  output: CreateAgentToolOutput,
): output is OperationStartedResponse {
  return (
    output.type === ResponseType.operation_started ||
    ("operation_id" in output && "tool_name" in output)
  );
}

export function isOperationPendingOutput(
  output: CreateAgentToolOutput,
): output is OperationPendingResponse {
  return output.type === ResponseType.operation_pending;
}

export function isOperationInProgressOutput(
  output: CreateAgentToolOutput,
): output is OperationInProgressResponse {
  return (
    output.type === ResponseType.operation_in_progress ||
    "tool_call_id" in output
  );
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
      return "Creating agent";
    case "input-available":
      return "Generating agent workflow";
    case "output-available": {
      const output = parseOutput(part.output);
      if (!output) return "Agent created";
      if (isOperationStartedOutput(output)) return "Agent creation started";
      if (isOperationPendingOutput(output)) return "Agent creation in progress";
      if (isOperationInProgressOutput(output))
        return "Agent creation already in progress";
      if (isAgentSavedOutput(output)) return `Saved: ${output.agent_name}`;
      if (isAgentPreviewOutput(output)) return `Preview: ${output.agent_name}`;
      if (isClarificationNeededOutput(output)) return "Needs clarification";
      return "Error creating agent";
    }
    case "output-error":
      return "Error creating agent";
    default:
      return "Processing";
  }
}

export function StateIcon({ state }: { state: ToolUIPart["state"] }) {
  switch (state) {
    case "input-streaming":
    case "input-available":
      return (
        <CircleNotchIcon
          className="h-4 w-4 animate-spin text-muted-foreground"
          weight="bold"
        />
      );
    case "output-available":
      return <CheckCircleIcon className="h-4 w-4 text-green-500" />;
    case "output-error":
      return <XCircleIcon className="h-4 w-4 text-red-500" />;
    default:
      return null;
  }
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
