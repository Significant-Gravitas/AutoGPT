import type { ToolUIPart } from "ai";
import {
  CheckCircleIcon,
  CircleNotchIcon,
  XCircleIcon,
} from "@phosphor-icons/react";

export interface ClarifyingQuestion {
  question: string;
  keyword: string;
  example?: string;
}

export interface OperationStartedOutput {
  type: "operation_started";
  message: string;
  session_id?: string;
  operation_id: string;
  tool_name: string;
}

export interface OperationPendingOutput {
  type: "operation_pending";
  message: string;
  session_id?: string;
  operation_id: string;
  tool_name: string;
}

export interface OperationInProgressOutput {
  type: "operation_in_progress";
  message: string;
  session_id?: string;
  tool_call_id: string;
}

export interface AgentPreviewOutput {
  type: "agent_preview";
  message: string;
  session_id?: string;
  agent_json: Record<string, unknown>;
  agent_name: string;
  description: string;
  node_count: number;
  link_count: number;
}

export interface AgentSavedOutput {
  type: "agent_saved";
  message: string;
  session_id?: string;
  agent_id: string;
  agent_name: string;
  library_agent_id: string;
  library_agent_link: string;
  agent_page_link: string;
}

export interface ClarificationNeededOutput {
  type: "clarification_needed";
  message: string;
  session_id?: string;
  questions: ClarifyingQuestion[];
}

export interface ErrorOutput {
  type: "error";
  message: string;
  session_id?: string;
  error?: string | null;
  details?: Record<string, unknown> | null;
}

export type CreateAgentToolOutput =
  | OperationStartedOutput
  | OperationPendingOutput
  | OperationInProgressOutput
  | AgentPreviewOutput
  | AgentSavedOutput
  | ClarificationNeededOutput
  | ErrorOutput;

function parseOutput(output: unknown): CreateAgentToolOutput | null {
  if (!output) return null;
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (!trimmed) return null;
    try {
      return JSON.parse(trimmed) as CreateAgentToolOutput;
    } catch {
      return null;
    }
  }
  if (typeof output === "object") return output as CreateAgentToolOutput;
  return null;
}

export function getCreateAgentToolOutput(
  part: unknown,
): CreateAgentToolOutput | null {
  if (!part || typeof part !== "object") return null;
  return parseOutput((part as { output?: unknown }).output);
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
      if (output.type === "operation_started") return "Agent creation started";
      if (output.type === "operation_pending")
        return "Agent creation in progress";
      if (output.type === "operation_in_progress")
        return "Agent creation already in progress";
      if (output.type === "agent_saved") return `Saved: ${output.agent_name}`;
      if (output.type === "agent_preview")
        return `Preview: ${output.agent_name}`;
      if (output.type === "clarification_needed") return "Needs clarification";
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
