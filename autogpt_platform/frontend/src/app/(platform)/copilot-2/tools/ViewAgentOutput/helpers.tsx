import type { ToolUIPart } from "ai";
import {
  CheckCircleIcon,
  CircleNotchIcon,
  XCircleIcon,
} from "@phosphor-icons/react";

export interface ViewAgentOutputInput {
  agent_name?: string;
  library_agent_id?: string;
  store_slug?: string;
  execution_id?: string;
  run_time?: string;
}

export interface ExecutionOutputInfo {
  execution_id: string;
  status: string;
  started_at?: string | null;
  ended_at?: string | null;
  outputs: Record<string, unknown[]>;
  inputs_summary?: Record<string, unknown> | null;
}

export interface AgentOutputOutput {
  type: "agent_output";
  message: string;
  session_id?: string;
  agent_name: string;
  agent_id: string;
  library_agent_id?: string | null;
  library_agent_link?: string | null;
  execution?: ExecutionOutputInfo | null;
  available_executions?: Array<Record<string, unknown>> | null;
  total_executions: number;
}

export interface NoResultsOutput {
  type: "no_results";
  message: string;
  session_id?: string;
  suggestions?: string[];
}

export interface ErrorOutput {
  type: "error";
  message: string;
  session_id?: string;
  error?: string | null;
  details?: Record<string, unknown> | null;
}

export type ViewAgentOutputToolOutput =
  | AgentOutputOutput
  | NoResultsOutput
  | ErrorOutput;

function parseOutput(output: unknown): ViewAgentOutputToolOutput | null {
  if (!output) return null;
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (!trimmed) return null;
    try {
      return JSON.parse(trimmed) as ViewAgentOutputToolOutput;
    } catch {
      return null;
    }
  }
  if (typeof output === "object") return output as ViewAgentOutputToolOutput;
  return null;
}

export function getViewAgentOutputToolOutput(
  part: unknown,
): ViewAgentOutputToolOutput | null {
  if (!part || typeof part !== "object") return null;
  return parseOutput((part as { output?: unknown }).output);
}

function getAgentIdentifierText(
  input: ViewAgentOutputInput | undefined,
): string | null {
  if (!input) return null;
  const libraryId = input.library_agent_id?.trim();
  if (libraryId) return `Library agent ${libraryId}`;
  const slug = input.store_slug?.trim();
  if (slug) return slug;
  const name = input.agent_name?.trim();
  if (name) return name;
  return null;
}

export function getAnimationText(part: {
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}): string {
  const input = part.input as ViewAgentOutputInput | undefined;
  const agent = getAgentIdentifierText(input);

  switch (part.state) {
    case "input-streaming":
      return "Looking up agent outputs";
    case "input-available":
      return agent ? `Loading outputs: ${agent}` : "Loading agent outputs";
    case "output-available": {
      const output = parseOutput(part.output);
      if (!output) return "Loaded agent outputs";
      if (output.type === "agent_output") {
        if (output.execution)
          return `Loaded output (${output.execution.status})`;
        return "Loaded agent outputs";
      }
      if (output.type === "no_results") return "No outputs found";
      return "Error loading agent output";
    }
    case "output-error":
      return "Error loading agent output";
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
