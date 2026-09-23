export interface CapabilityListing {
  id: string;
  name: string;
  purpose: string;
  kind: string;
  class?: string;
  connected?: boolean | null;
  eager?: boolean;
}

export interface CapabilityListOutput {
  type: "capability_list";
  query: string;
  capabilities: CapabilityListing[];
  count: number;
  fallback?: CapabilityListing[];
  service?: string | null;
  message?: string;
}

export interface FindCapabilityInput {
  query?: string;
  context?: string;
  kind?: string;
}

export interface FindCapabilityToolPart {
  type: string;
  toolCallId: string;
  state: string;
  input?: unknown;
  output?: unknown;
}

export function parseOutput(output: unknown): CapabilityListOutput | null {
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
    const candidate = output as { type?: unknown; capabilities?: unknown };
    if (
      candidate.type === "capability_list" &&
      Array.isArray(candidate.capabilities)
    ) {
      return output as CapabilityListOutput;
    }
  }
  return null;
}

export function queryOf(part: FindCapabilityToolPart): string | undefined {
  // `input` is whatever the model streamed, so `query` is not necessarily a
  // string -- and this runs while the call is still streaming.
  const query = (part.input as FindCapabilityInput | undefined)?.query;
  return typeof query === "string" ? query.trim() : undefined;
}

export function getAnimationText(part: FindCapabilityToolPart): string {
  const query = queryOf(part);
  const queryText = query ? ` for "${query}"` : "";
  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return `Searching capabilities${queryText}`;
    case "output-available": {
      const parsed = parseOutput(part.output);
      if (parsed) {
        return `Found ${parsed.count} capabilit${parsed.count === 1 ? "y" : "ies"}${queryText}`;
      }
      return `Searched capabilities${queryText}`;
    }
    case "output-error":
      return `Search failed${queryText}`;
    default:
      return "Searching capabilities";
  }
}

export function kindLabel(item: CapabilityListing): string {
  if (item.kind === "mcp_server") return "integration";
  if (item.kind === "skill") return "skill";
  if (item.class === "primitive") return "building block";
  if (item.kind === "tool") return "tool";
  return "action";
}
