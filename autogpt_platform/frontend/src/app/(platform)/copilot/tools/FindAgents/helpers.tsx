import type { AgentInfo } from "@/app/api/__generated__/models/agentInfo";
import type { AgentsFoundResponse } from "@/app/api/__generated__/models/agentsFoundResponse";
import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import type { NoResultsResponse } from "@/app/api/__generated__/models/noResultsResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import {
  FolderOpenIcon,
  MagnifyingGlassIcon,
  SquaresFourIcon,
  StorefrontIcon,
} from "@phosphor-icons/react";
import { ToolUIPart } from "ai";

export interface FindAgentInput {
  query: string;
}

export type FindAgentsOutput =
  | AgentsFoundResponse
  | NoResultsResponse
  | ErrorResponse;

export type FindAgentsToolType =
  | "tool-find_agent"
  | "tool-find_library_agent"
  | (string & {});

function parseOutput(output: unknown): FindAgentsOutput | null {
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
      type === ResponseType.agents_found ||
      type === ResponseType.no_results ||
      type === ResponseType.error
    ) {
      return output as FindAgentsOutput;
    }
    if ("agents" in output && "count" in output)
      return output as AgentsFoundResponse;
    if ("suggestions" in output && !("error" in output))
      return output as NoResultsResponse;
    if ("error" in output || "details" in output)
      return output as ErrorResponse;
  }
  return null;
}

export function getFindAgentsOutput(part: unknown): FindAgentsOutput | null {
  if (!part || typeof part !== "object") return null;
  return parseOutput((part as { output?: unknown }).output);
}

export function isAgentsFoundOutput(
  output: FindAgentsOutput,
): output is AgentsFoundResponse {
  return output.type === ResponseType.agents_found || "agents" in output;
}

export function isNoResultsOutput(
  output: FindAgentsOutput,
): output is NoResultsResponse {
  return (
    output.type === ResponseType.no_results ||
    ("suggestions" in output && !("error" in output))
  );
}

export function isErrorOutput(
  output: FindAgentsOutput,
): output is ErrorResponse {
  return output.type === ResponseType.error || "error" in output;
}

export function getSourceLabelFromToolType(toolType?: FindAgentsToolType): {
  source: "marketplace" | "library" | "unknown";
  label: string;
} {
  if (toolType === "tool-find_library_agent") {
    return { source: "library", label: "Library" };
  }
  if (toolType === "tool-find_agent") {
    return { source: "marketplace", label: "Marketplace" };
  }
  return { source: "unknown", label: "Agents" };
}

export function getAnimationText(part: {
  type?: FindAgentsToolType;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}): string {
  const { source } = getSourceLabelFromToolType(part.type);
  const query = (part.input as FindAgentInput | undefined)?.query?.trim();

  // Action phrase matching legacy ToolCallMessage
  const actionPhrase =
    source === "library"
      ? "Looking for library agents"
      : "Looking for agents in the marketplace";

  const queryText = query ? ` matching "${query}"` : "";

  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return `${actionPhrase}${queryText}`;

    case "output-available": {
      const output = parseOutput(part.output);
      if (!output) {
        return `${actionPhrase}${queryText}`;
      }
      if (isNoResultsOutput(output)) {
        return `No agents found${queryText}`;
      }
      if (isAgentsFoundOutput(output)) {
        const count = output.count ?? output.agents?.length ?? 0;
        return `Found ${count} agent${count === 1 ? "" : "s"}${queryText}`;
      }
      if (isErrorOutput(output)) {
        return `Error finding agents${queryText}`;
      }
      return `${actionPhrase}${queryText}`;
    }

    case "output-error":
      return `Error finding agents${queryText}`;

    default:
      return actionPhrase;
  }
}

export function getAgentHref(agent: AgentInfo): string | null {
  if (agent.source === "library") {
    return `/library/agents/${encodeURIComponent(agent.id)}`;
  }

  const [creator, slug, ...rest] = agent.id.split("/");
  if (!creator || !slug || rest.length > 0) return null;
  return `/marketplace/agent/${encodeURIComponent(creator)}/${encodeURIComponent(slug)}`;
}

export function ToolIcon({
  toolType,
  isStreaming,
  isError,
}: {
  toolType?: FindAgentsToolType;
  isStreaming?: boolean;
  isError?: boolean;
}) {
  const { source } = getSourceLabelFromToolType(toolType);
  const IconComponent =
    source === "library" ? MagnifyingGlassIcon : SquaresFourIcon;

  return (
    <IconComponent
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

export function AccordionIcon({ toolType }: { toolType?: FindAgentsToolType }) {
  const { source } = getSourceLabelFromToolType(toolType);
  const IconComponent = source === "library" ? FolderOpenIcon : StorefrontIcon;
  return <IconComponent size={32} weight="light" />;
}
