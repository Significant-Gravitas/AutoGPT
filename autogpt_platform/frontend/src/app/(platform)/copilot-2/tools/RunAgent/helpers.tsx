import type { ToolUIPart } from "ai";
import {
  CheckCircleIcon,
  CircleNotchIcon,
  XCircleIcon,
} from "@phosphor-icons/react";

export interface RunAgentInput {
  username_agent_slug?: string;
  library_agent_id?: string;
  inputs?: Record<string, unknown>;
  use_defaults?: boolean;
  schedule_name?: string;
  cron?: string;
  timezone?: string;
}

export interface CredentialsMeta {
  id: string;
  provider: string;
  type: string;
  title: string;
}

export interface SetupInfo {
  agent_id: string;
  agent_name: string;
  requirements: {
    credentials: CredentialsMeta[];
    inputs: Array<{
      name: string;
      title: string;
      type: string;
      description: string;
      required: boolean;
    }>;
    execution_modes: string[];
  };
  user_readiness: {
    has_all_credentials: boolean;
    missing_credentials: Record<string, CredentialsMeta>;
    ready_to_run: boolean;
  };
}

export interface SetupRequirementsOutput {
  type: "setup_requirements";
  message: string;
  session_id?: string;
  setup_info: SetupInfo;
  graph_id?: string | null;
  graph_version?: number | null;
}

export interface ExecutionStartedOutput {
  type: "execution_started";
  message: string;
  session_id?: string;
  execution_id: string;
  graph_id: string;
  graph_name: string;
  library_agent_id?: string | null;
  library_agent_link?: string | null;
  status: string;
}

export interface ErrorOutput {
  type: "error";
  message: string;
  session_id?: string;
  error?: string | null;
  details?: Record<string, unknown> | null;
}

export interface NeedLoginOutput {
  type: "need_login";
  message: string;
  session_id?: string;
}

export type RunAgentToolOutput =
  | SetupRequirementsOutput
  | ExecutionStartedOutput
  | NeedLoginOutput
  | ErrorOutput;

function parseOutput(output: unknown): RunAgentToolOutput | null {
  if (!output) return null;
  if (typeof output === "string") {
    const trimmed = output.trim();
    if (!trimmed) return null;
    try {
      return JSON.parse(trimmed) as RunAgentToolOutput;
    } catch {
      return null;
    }
  }
  if (typeof output === "object") return output as RunAgentToolOutput;
  return null;
}

export function getRunAgentToolOutput(
  part: unknown,
): RunAgentToolOutput | null {
  if (!part || typeof part !== "object") return null;
  return parseOutput((part as { output?: unknown }).output);
}

function getAgentIdentifierText(
  input: RunAgentInput | undefined,
): string | null {
  if (!input) return null;
  const slug = input.username_agent_slug?.trim();
  if (slug) return slug;
  const libraryId = input.library_agent_id?.trim();
  if (libraryId) return `Library agent ${libraryId}`;
  return null;
}

function getExecutionModeText(input: RunAgentInput | undefined): string | null {
  if (!input) return null;
  const isSchedule = Boolean(input.schedule_name?.trim() || input.cron?.trim());
  return isSchedule ? "Scheduled run" : "Run";
}

export function getAnimationText(part: {
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}): string {
  const input = part.input as RunAgentInput | undefined;
  const agentIdentifier = getAgentIdentifierText(input);
  const mode = getExecutionModeText(input);

  switch (part.state) {
    case "input-streaming":
      return "Preparing to run agent";
    case "input-available":
      return agentIdentifier ? `${mode}: ${agentIdentifier}` : "Running agent";
    case "output-available": {
      const output = parseOutput(part.output);
      if (!output) return "Agent run updated";
      if (output.type === "execution_started") {
        return `Started: ${output.graph_name}`;
      }
      if (output.type === "setup_requirements") {
        return `Needs setup: ${output.setup_info.agent_name}`;
      }
      if (output.type === "need_login") return "Sign in required to run agent";
      return "Error running agent";
    }
    case "output-error":
      return "Error running agent";
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
