import { beautifyString } from "@/lib/utils";
import type { HeldOutcome } from "../ChatMessagesContainer/heldCallRows";
import type { ChainRow } from "./helpers";
import { asObject } from "./resultHelpers";
import { askText, getAskLabel } from "./toolCatalog.ask";
import { getCatalogLabel } from "./toolCatalog";

export type HeldState =
  | "waiting"
  | "approved"
  | "rejected"
  | "expired"
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
  const reviewId = typeof data.review_id === "string" ? data.review_id : null;
  const tool = heldToolName(data, row.tool);
  const ask = heldAskText(tool, row.input);
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
  switch (outcome.outcome) {
    case "approved":
      return {
        ...settle(
          row,
          getCatalogLabel(tool, row.input, "done")?.text ?? ask,
          "approved",
          reviewId,
        ),
        output: outcome.output,
      };
    case "rejected":
      return settle(row, didnt, "rejected", reviewId);
    case "expired":
      return settle(row, didnt, "expired", reviewId);
    default:
      return settle(row, didnt, "not-run", reviewId);
  }
}

// The held call's own tool, which a capability row names only in its output.
export function heldToolName(
  output: Record<string, unknown>,
  fallback: string,
) {
  const name = output.tool_name;
  return typeof name === "string" && name.trim() ? name.trim() : fallback;
}

export function heldAskText(toolName: string, input: unknown) {
  const label = getAskLabel(toolName, asObject(input) ?? {});
  return label
    ? askText(label)
    : `Run ${beautifyString(toolName.replace(/^run_/, "")).toLowerCase()}`;
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
