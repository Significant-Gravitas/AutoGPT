import type { AgentOutputResponse } from "@/app/api/__generated__/models/agentOutputResponse";
import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import type { NoResultsResponse } from "@/app/api/__generated__/models/noResultsResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import { EyeIcon, MonitorIcon } from "@phosphor-icons/react";
import type { ToolUIPart } from "ai";

export interface ViewAgentOutputInput {
  agent_name?: string;
  library_agent_id?: string;
  store_slug?: string;
  execution_id?: string;
  run_time?: string;
}

export type ViewAgentOutputToolOutput =
  | AgentOutputResponse
  | NoResultsResponse
  | ErrorResponse;

function parseOutput(output: unknown): ViewAgentOutputToolOutput | null {
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
      type === ResponseType.agent_output ||
      type === ResponseType.no_results ||
      type === ResponseType.error
    ) {
      return output as ViewAgentOutputToolOutput;
    }
    if ("agent_id" in output && "agent_name" in output) {
      return output as AgentOutputResponse;
    }
    if ("suggestions" in output && !("error" in output)) {
      return output as NoResultsResponse;
    }
    if ("error" in output || "details" in output)
      return output as ErrorResponse;
  }
  return null;
}

export function isAgentOutputResponse(
  output: ViewAgentOutputToolOutput,
): output is AgentOutputResponse {
  return output.type === ResponseType.agent_output || "agent_id" in output;
}

export function isNoResultsResponse(
  output: ViewAgentOutputToolOutput,
): output is NoResultsResponse {
  return (
    output.type === ResponseType.no_results ||
    ("suggestions" in output && !("error" in output))
  );
}

export function isErrorResponse(
  output: ViewAgentOutputToolOutput,
): output is ErrorResponse {
  return output.type === ResponseType.error || "error" in output;
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
  const agentText = agent ? ` "${agent}"` : "";

  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return `Retrieving agent output${agentText}`;
    case "output-available": {
      const output = parseOutput(part.output);
      if (!output) return `Retrieving agent output${agentText}`;
      if (isAgentOutputResponse(output)) {
        if (output.execution)
          return `Retrieved output (${output.execution.status})`;
        return "Retrieved agent output";
      }
      if (isNoResultsResponse(output)) return "No outputs found";
      return "Error loading agent output";
    }
    case "output-error":
      return "Error loading agent output";
    default:
      return "Retrieving agent output";
  }
}

export function ToolIcon({
  isStreaming,
  isError,
}: {
  isStreaming?: boolean;
  isError?: boolean;
}) {
  return (
    <EyeIcon
      size={14}
      weight="regular"
      className={
        isError
          ? "text-red-500"
          : isStreaming
            ? "text-neutral-500"
            : "text-neutral-400"
      }
    />
  );
}

export function AccordionIcon() {
  return <MonitorIcon size={32} weight="light" />;
}

export function formatMaybeJson(value: unknown): string {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}
