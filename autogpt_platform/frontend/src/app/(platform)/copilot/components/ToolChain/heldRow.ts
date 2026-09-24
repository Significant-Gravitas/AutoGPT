import type { HeldOutcome } from "../ChatMessagesContainer/heldCallRows";
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
  | "not-run";

export interface HeldRowInfo {
  state: HeldState;
  reviewId: string | null;
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
  const didnt = `Didn't ${lowerFirst(ask)}`;
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
      held: { state: "waiting", reviewId },
    };
  }
  if (outcome.outcome !== "approved") {
    return settle(row, didnt, outcome.outcome, reviewId);
  }
  const result = asObject(outcome.output);
  const done = getCatalogLabel(tool, row.input, "done")?.text ?? ask;
  // It ran and failed: the normal error row, still marked as approved.
  if (result?.type === "error") {
    return {
      ...settle(row, done, "approved", reviewId),
      state: "error",
      detail: str(result, "message", "error") ?? undefined,
      output: outcome.output,
    };
  }
  return { ...settle(row, done, "approved", reviewId), output: outcome.output };
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
): ChainRow {
  return { ...row, text, requiresAction: false, held: { state, reviewId } };
}

function lowerFirst(text: string) {
  return text.charAt(0).toLowerCase() + text.slice(1);
}
