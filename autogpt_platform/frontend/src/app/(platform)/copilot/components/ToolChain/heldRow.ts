import type { HeldOutcome } from "../ChatMessagesContainer/heldCallRows";
import { COPILOT_GATE_NODE_PREFIX } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { fallbackAsk } from "../ApprovalQueue/helpers";
import type { ChainRow } from "./helpers";
import { asObject, str } from "./resultHelpers";
import { getCatalogLabel } from "./toolCatalog";

export type HeldState =
  | "waiting"
  | "approved"
  | "rejected"
  | "expired"
  | "closed"
  | "unknown"
  | "not-run";

export interface HeldRowInfo {
  state: HeldState;
  reviewId: string | null;
  // A read held for carrying instructions, not an action held for approval.
  read?: boolean;
}

// A held call's row names the action while it waits, then what became of it.
export function applyHeldOutcome(
  row: ChainRow,
  outcomes: ReadonlyMap<string, HeldOutcome>,
): ChainRow {
  const data = asObject(row.output);
  if (!row.tool || data?.type !== "approval_required") return row;
  const reviewId = str(data, "review_id");
  const tool = heldToolName(data, row.tool);
  const ask = heldAskText(data, tool);
  const read = isHeldReadId(reviewId);
  const didnt = read ? ask : `Didn't ${lowerFirst(ask)}`;
  if (!reviewId) {
    return settle(
      row,
      `Couldn't ask about ${lowerFirst(ask)}`,
      "not-run",
      reviewId,
    );
  }
  const outcome = outcomes.get(row.key);
  if (!outcome) {
    return {
      ...row,
      text: ask,
      requiresAction: true,
      held: { state: "waiting", reviewId, read },
    };
  }
  // It may have run; claim neither success nor a refusal.
  if (outcome.outcome === "unknown") {
    return settle(row, ask, "unknown", reviewId);
  }
  if (outcome.outcome !== "approved") {
    return settle(row, didnt, outcome.outcome, reviewId, read);
  }
  const result = asObject(outcome.output);
  const done = getCatalogLabel(tool, row.input, "done")?.text ?? ask;
  // It ran and failed: the normal error row, still marked as approved.
  if (result?.type === "error") {
    return {
      ...settle(row, done, "approved", reviewId, read),
      state: "error",
      detail: str(result, "message", "error") ?? undefined,
      output: outcome.output,
    };
  }
  return {
    ...settle(row, done, "approved", reviewId, read),
    output: outcome.output,
  };
}

// The held call's own tool, which a capability row names only in its output.
export function heldToolName(
  output: Record<string, unknown>,
  fallback: string,
) {
  return str(output, "tool_name")?.trim() ?? fallback;
}

// The server's words for the call, quoted as every other chain row quotes.
export function heldAskText(output: Record<string, unknown>, toolName: string) {
  const ask = str(output, "ask") ?? fallbackAsk(toolName);
  const object = str(output, "object");
  return object ? `${ask} "${object}"` : ask;
}

function settle(
  row: ChainRow,
  text: string,
  state: HeldState,
  reviewId: string | null,
  read = false,
): ChainRow {
  return {
    ...row,
    text,
    requiresAction: false,
    held: { state, reviewId, read },
  };
}

function isHeldReadId(reviewId: string | null) {
  return !!reviewId?.startsWith(`${COPILOT_GATE_NODE_PREFIX}read-`);
}

function lowerFirst(text: string) {
  return text.charAt(0).toLowerCase() + text.slice(1);
}
