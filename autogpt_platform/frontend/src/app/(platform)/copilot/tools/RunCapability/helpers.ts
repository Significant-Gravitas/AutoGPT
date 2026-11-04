import type { ToolUIPart } from "ai";

export interface RunCapabilityInput {
  id?: string;
  review_id?: string;
  input?: Record<string, unknown>;
  validate_only?: boolean;
  expand?: boolean;
}

const MCP_OUTPUT_TYPES = new Set(["mcp_tools_discovered", "mcp_tool_output"]);

// Review ids are "<prefix><target>:<random hex>", and the prefix says which
// renderer the paused call belongs to.
const MCP_REVIEW_PREFIX = "copilot-mcp-";
const BLOCK_REVIEW_PREFIX = "copilot-node-";

export function parseOutputObject(
  output: unknown,
): Record<string, unknown> | null {
  if (!output) return null;
  if (typeof output === "string") {
    try {
      return parseOutputObject(JSON.parse(output) as unknown);
    } catch {
      return null;
    }
  }
  return typeof output === "object"
    ? (output as Record<string, unknown>)
    : null;
}

/** `resume_capability` names its target by `review_id`, not `id`. Reading the
 *  target back out of the review id is what keeps a resumed MCP call from
 *  rendering as a nameless block for the seconds before its output lands. */
export function capabilityId(input: unknown): string {
  const call = input as RunCapabilityInput | undefined;
  const id = typeof call?.id === "string" ? call.id.trim() : "";
  if (id) return id;
  const reviewId =
    typeof call?.review_id === "string" ? call.review_id.trim() : "";
  const separator = reviewId.lastIndexOf(":");
  if (separator < 0) return "";
  const target = reviewId.slice(0, separator);
  for (const [prefix, kind] of [
    [MCP_REVIEW_PREFIX, "mcp"],
    [BLOCK_REVIEW_PREFIX, "block"],
  ] as const) {
    if (!target.startsWith(prefix)) continue;
    const name = target.slice(prefix.length);
    return name ? `${kind}:${name}` : "";
  }
  return "";
}

export function isMcpCapability(
  id: string,
  output: Record<string, unknown> | null,
): boolean {
  if (id.startsWith("mcp:") || id.startsWith("https://")) return true;
  if (!output) return false;
  const type = output.type;
  return (
    (typeof type === "string" && MCP_OUTPUT_TYPES.has(type)) ||
    "server_url" in output
  );
}

export function isCapabilityDetails(
  output: Record<string, unknown> | null,
): boolean {
  return output?.type === "capability_details";
}

function stripPrefix(id: string, prefix: string): string {
  return id.startsWith(prefix) ? id.slice(prefix.length) : id;
}

/** Re-shape a run_capability part into the legacy run_block part shape so
 *  the block renderer (details, output, review, setup card) is reused. */
export function asBlockPart(part: ToolUIPart): ToolUIPart {
  const input = (part.input ?? {}) as RunCapabilityInput;
  return {
    ...part,
    type: "tool-run_block",
    input: {
      block_id: stripPrefix(capabilityId(input), "block:"),
      input_data: input.input ?? {},
      validate_only: input.validate_only ?? false,
    },
  } as ToolUIPart;
}

/** Re-shape a run_capability part into the legacy run_mcp_tool part shape. */
export function asMcpPart(
  part: ToolUIPart,
  output: Record<string, unknown> | null,
): ToolUIPart {
  const input = (part.input ?? {}) as RunCapabilityInput;
  const id = capabilityId(input);
  const serverUrl =
    typeof output?.server_url === "string"
      ? output.server_url
      : id.startsWith("https://")
        ? id
        : "";
  const inner = input.input ?? {};
  return {
    ...part,
    type: "tool-run_mcp_tool",
    input: {
      server_url: serverUrl,
      tool_name: typeof inner.tool === "string" ? inner.tool : undefined,
      tool_arguments: inner.arguments,
      surface_connect_card: inner.connect === true,
    },
  } as ToolUIPart;
}
