import { ToolUIPart } from "ai";
import {
  CheckCircleIcon,
  CircleNotchIcon,
  XCircleIcon,
} from "@phosphor-icons/react";

export interface FindAgentInput {
  query: string;
}

export interface AgentInfo {
  id: string;
  name: string;
  description: string;
  source?: "marketplace" | "library" | string;
}

export interface AgentsFoundOutput {
  type: "agents_found";
  title?: string;
  message?: string;
  session_id?: string;
  agents: AgentInfo[];
  count: number;
}

export interface NoResultsOutput {
  type: "no_results";
  message: string;
  suggestions?: string[];
  session_id?: string;
}

export interface ErrorOutput {
  type: "error";
  message: string;
  error?: string;
  session_id?: string;
}

export type FindAgentsOutput =
  | AgentsFoundOutput
  | NoResultsOutput
  | ErrorOutput;

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
      return JSON.parse(trimmed) as FindAgentsOutput;
    } catch {
      return null;
    }
  }
  if (typeof output === "object") {
    return output as FindAgentsOutput;
  }
  return null;
}

export function getFindAgentsOutput(part: unknown): FindAgentsOutput | null {
  if (!part || typeof part !== "object") return null;
  return parseOutput((part as { output?: unknown }).output);
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
      if (output.type === "no_results") {
        return query
          ? `No agents found ${scope} for "${query}"`
          : `No agents found ${scope}`;
      }
      if (output.type === "agents_found") {
        const count = output.count ?? output.agents?.length ?? 0;
        const countText = `Found ${count} agent${count === 1 ? "" : "s"}`;
        if (query) return `${countText} ${scope} for "${query}"`;
        return `${countText} ${scope}`;
      }
      if (output.type === "error") {
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
