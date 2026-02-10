import type { AgentDetailsResponse } from "@/app/api/__generated__/models/agentDetailsResponse";
import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import type { ExecutionStartedResponse } from "@/app/api/__generated__/models/executionStartedResponse";
import type { NeedLoginResponse } from "@/app/api/__generated__/models/needLoginResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import {
  PlayIcon,
  RocketLaunchIcon,
  WarningDiamondIcon,
} from "@phosphor-icons/react";
import type { ToolUIPart } from "ai";
import { OrbitLoader } from "../../components/OrbitLoader/OrbitLoader";

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

export function getAnimationText(part: {
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}): string {
  const input = part.input as RunAgentInput | undefined;
  const agentIdentifier = getAgentIdentifierText(input);
  const isSchedule = Boolean(
    input?.schedule_name?.trim() || input?.cron?.trim(),
  );
  const actionPhrase = isSchedule
    ? "Scheduling the agent to run"
    : "Running the agent";
  const identifierText = agentIdentifier ? ` "${agentIdentifier}"` : "";

  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return `${actionPhrase}${identifierText}`;
    case "output-available": {
      const output = parseOutput(part.output);
      if (!output) return `${actionPhrase}${identifierText}`;
      if (isRunAgentExecutionStartedOutput(output)) {
        return `Started "${output.graph_name}"`;
      }
      if (isRunAgentAgentDetailsOutput(output)) {
        return `Agent inputs needed for "${output.agent.name}"`;
      }
      if (isRunAgentSetupRequirementsOutput(output)) {
        return `Setup needed for "${output.setup_info.agent_name}"`;
      }
      if (isRunAgentNeedLoginOutput(output))
        return "Sign in required to run agent";
      return "Error running agent";
    }
    case "output-error":
      return "Error running agent";
    default:
      return actionPhrase;
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
  return <PlayIcon size={14} weight="regular" className="text-neutral-400" />;
}

export function AccordionIcon() {
  return <RocketLaunchIcon size={28} weight="light" />;
}

export function formatMaybeJson(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function getAccordionMeta(output: RunAgentToolOutput): {
  icon: React.ReactNode;
  title: string;
  titleClassName?: string;
  description?: string;
} {
  const icon = <AccordionIcon />;

  if (isRunAgentExecutionStartedOutput(output)) {
    const statusText =
      typeof output.status === "string" && output.status.trim()
        ? output.status.trim()
        : "started";
    return {
      icon: <OrbitLoader size={28} className="text-neutral-700" />,
      title: output.graph_name,
      description: `Status: ${statusText}`,
    };
  }

  if (isRunAgentAgentDetailsOutput(output)) {
    return {
      icon,
      title: output.agent.name,
      description: "Inputs required",
    };
  }

  if (isRunAgentSetupRequirementsOutput(output)) {
    const missingCredsCount = Object.keys(
      (output.setup_info.user_readiness?.missing_credentials ?? {}) as Record<
        string,
        unknown
      >,
    ).length;
    return {
      icon,
      title: output.setup_info.agent_name,
      description:
        missingCredsCount > 0
          ? `Missing ${missingCredsCount} credential${missingCredsCount === 1 ? "" : "s"}`
          : output.message,
    };
  }

  if (isRunAgentNeedLoginOutput(output)) {
    return { icon, title: "Sign in required" };
  }

  return {
    icon: (
      <WarningDiamondIcon size={28} weight="light" className="text-red-500" />
    ),
    title: "Error",
    titleClassName: "text-red-500",
  };
}
