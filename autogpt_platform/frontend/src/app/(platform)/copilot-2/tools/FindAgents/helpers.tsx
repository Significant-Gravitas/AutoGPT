import { ToolUIPart } from "ai";
import {
  CheckCircleIcon,
  CircleNotchIcon,
  XCircleIcon,
} from "@phosphor-icons/react";
import type { AgentInfo } from "@/app/api/__generated__/models/agentInfo";
import type { AgentsFoundResponse } from "@/app/api/__generated__/models/agentsFoundResponse";
import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import type { NoResultsResponse } from "@/app/api/__generated__/models/noResultsResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";

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
  const { label, source } = getSourceLabelFromToolType(part.type);
  switch (part.state) {
    case "input-streaming":
      return `Searching ${label.toLowerCase()} agents for you`;

    case "input-available": {
      const query = (part.input as FindAgentInput | undefined)?.query?.trim();
      if (query) {
        return source === "library"
          ? `Finding library agents matching "${query}"`
          : `Finding marketplace agents matching "${query}"`;
      }
      return source === "library" ? "Finding library agents" : "Finding agents";
    }

    case "output-available": {
      const output = parseOutput(part.output);
      const query = (part.input as FindAgentInput | undefined)?.query?.trim();
      const scope = source === "library" ? "in your library" : "in marketplace";
      if (!output) {
        return query ? `Found agents ${scope} for "${query}"` : "Found agents";
      }
      if (isNoResultsOutput(output)) {
        return query
          ? `No agents found ${scope} for "${query}"`
          : `No agents found ${scope}`;
      }
      if (isAgentsFoundOutput(output)) {
        const count = output.count ?? output.agents?.length ?? 0;
        const countText = `Found ${count} agent${count === 1 ? "" : "s"}`;
        if (query) return `${countText} ${scope} for "${query}"`;
        return `${countText} ${scope}`;
      }
      if (isErrorOutput(output)) {
        return `Error finding agents ${scope}`;
      }
      return `Found agents ${scope}`;
    }

    case "output-error":
      return source === "library"
        ? "Error finding agents in your library"
        : "Error finding agents in marketplace";

    default:
      return "Processing";
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
