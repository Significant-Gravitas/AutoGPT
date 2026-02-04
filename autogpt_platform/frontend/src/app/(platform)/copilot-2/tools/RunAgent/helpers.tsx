import type { ToolUIPart } from "ai";
import {
  CheckCircleIcon,
  CircleNotchIcon,
  XCircleIcon,
} from "@phosphor-icons/react";
import type { AgentDetailsResponse } from "@/app/api/__generated__/models/agentDetailsResponse";
import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import type { ExecutionStartedResponse } from "@/app/api/__generated__/models/executionStartedResponse";
import type { NeedLoginResponse } from "@/app/api/__generated__/models/needLoginResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";

export interface RunAgentInput {
  username_agent_slug?: string;
  library_agent_id?: string;
  inputs?: Record<string, unknown>;
  use_defaults?: boolean;
  schedule_name?: string;
  cron?: string;
  timezone?: string;
}

export type RunAgentToolOutput =
  | SetupRequirementsResponse
  | ExecutionStartedResponse
  | AgentDetailsResponse
  | NeedLoginResponse
  | ErrorResponse;

const RUN_AGENT_OUTPUT_TYPES = new Set<string>([
  ResponseType.setup_requirements,
  ResponseType.execution_started,
  ResponseType.agent_details,
  ResponseType.need_login,
  ResponseType.error,
]);

export function isRunAgentSetupRequirementsOutput(
  output: RunAgentToolOutput,
): output is SetupRequirementsResponse {
  return (
    output.type === ResponseType.setup_requirements ||
    ("setup_info" in output && typeof output.setup_info === "object")
  );
}

export function isRunAgentExecutionStartedOutput(
  output: RunAgentToolOutput,
): output is ExecutionStartedResponse {
  return (
    output.type === ResponseType.execution_started || "execution_id" in output
  );
}

export function isRunAgentAgentDetailsOutput(
  output: RunAgentToolOutput,
): output is AgentDetailsResponse {
  return output.type === ResponseType.agent_details || "agent" in output;
}

export function isRunAgentNeedLoginOutput(
  output: RunAgentToolOutput,
): output is NeedLoginResponse {
  return output.type === ResponseType.need_login;
}

export function isRunAgentErrorOutput(
  output: RunAgentToolOutput,
): output is ErrorResponse {
  return output.type === ResponseType.error || "error" in output;
}

function parseOutput(output: unknown): RunAgentToolOutput | null {
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
    if (typeof type === "string" && RUN_AGENT_OUTPUT_TYPES.has(type)) {
      return output as RunAgentToolOutput;
    }
    if ("execution_id" in output) return output as ExecutionStartedResponse;
    if ("setup_info" in output) return output as SetupRequirementsResponse;
    if ("agent" in output) return output as AgentDetailsResponse;
    if ("error" in output || "details" in output)
      return output as ErrorResponse;
    if (type === ResponseType.need_login) return output as NeedLoginResponse;
  }
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
      if (isRunAgentExecutionStartedOutput(output)) {
        return `Started: ${output.graph_name}`;
      }
      if (isRunAgentAgentDetailsOutput(output)) {
        return `Agent inputs: ${output.agent.name}`;
      }
      if (isRunAgentSetupRequirementsOutput(output)) {
        return `Needs setup: ${output.setup_info.agent_name}`;
      }
      if (isRunAgentNeedLoginOutput(output))
        return "Sign in required to run agent";
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
