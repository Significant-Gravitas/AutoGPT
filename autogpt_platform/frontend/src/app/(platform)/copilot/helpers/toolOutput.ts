import * as Sentry from "@sentry/nextjs";

/**
 * Tool types whose JSON output can surface interactive cards in chat
 * (setup_requirements sign-in cards, pickers, review prompts). When their
 * output arrives corrupted the user loses a card they cannot recover —
 * unlike read-only tools where a broken payload only degrades a summary.
 */
export const CARD_CAPABLE_TOOL_TYPES: ReadonlySet<string> = new Set([
  "tool-run_block",
  "tool-continue_run_block",
  "tool-connect_integration",
  "tool-run_agent",
  "tool-run_mcp_tool",
]);

/**
 * True when `output` looks like a JSON document but doesn't parse — the
 * signature of a payload truncated or mangled in transit. Plain-text
 * outputs (error strings, file-reference notes) are not flagged: only
 * strings that start as JSON and fail to parse.
 */
export function isUnparseableJsonOutput(output: unknown): boolean {
  if (typeof output !== "string") return false;
  const trimmed = output.trim();
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return false;
  try {
    JSON.parse(trimmed);
    return false;
  } catch {
    return true;
  }
}

export function isCorruptedCardToolPart(part: {
  type: string;
  state?: unknown;
  output?: unknown;
}): boolean {
  if (!CARD_CAPABLE_TOOL_TYPES.has(part.type)) return false;
  if (part.state !== "output-available") return false;
  return isUnparseableJsonOutput(part.output);
}

/**
 * Bounded so the dedupe set can't grow for the tab's lifetime and so a
 * `toolCallId` reused in a much later conversation eventually evicts its old
 * entry and gets reported again instead of being silently dropped.
 */
const MAX_REPORTED_TOOL_CALL_IDS = 500;
const reportedToolCallIds = new Set<string>();

/**
 * Report a corrupted tool output to Sentry, once per toolCallId — safe to
 * call from render since repeated invocations are no-ops.
 */
export function reportCorruptedToolOutput(
  toolCallId: string,
  toolType: string,
): void {
  if (!toolCallId || reportedToolCallIds.has(toolCallId)) return;
  if (reportedToolCallIds.size >= MAX_REPORTED_TOOL_CALL_IDS) {
    const oldest = reportedToolCallIds.values().next().value;
    if (oldest !== undefined) reportedToolCallIds.delete(oldest);
  }
  reportedToolCallIds.add(toolCallId);
  Sentry.captureMessage(
    "Copilot tool output unparseable — interactive card lost",
    {
      level: "warning",
      tags: { context: "copilot-tool-output", toolType },
      extra: { toolCallId },
    },
  );
}
