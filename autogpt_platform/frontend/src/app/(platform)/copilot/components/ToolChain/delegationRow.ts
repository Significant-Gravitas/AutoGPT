import type { ChatDelegation, LiveDelegationStatus } from "../../delegations";
import { isTurnedDown } from "../../delegationOutput";
import type { ChainRow } from "./helpers";

const HANDOFF_TOOLS = new Set(["delegate_to_expert", "handoff_to_expert"]);

/** A hand-off to a teammate renders no card under its row: the status line
 *  and the nodes on the wire tell it. A result poll of a teammate's run is
 *  the same story, told by the same line. */
export function isDelegatedRow(row: ChainRow): boolean {
  if (!row.tool) return false;
  if (HANDOFF_TOOLS.has(row.tool)) return true;
  const output =
    row.output && typeof row.output === "object"
      ? (row.output as { expert?: unknown })
      : null;
  return row.tool === "get_sub_session_result" && !!output?.expert;
}

export type HandoffNode = "loading" | "question" | "answered" | null;

/** Ties a hand-off row to its delegation. A teammate waiting on the user
 *  keeps its row on screen when the chain collapses, like any action row. */
export function withDelegation(
  row: ChainRow,
  delegation: ChatDelegation | undefined,
  liveStatus: LiveDelegationStatus | undefined,
  readOnly: boolean,
): ChainRow {
  if (!delegation || row.held?.state === "waiting") return row;
  const status = liveStatus ?? delegation.status;
  return {
    ...row,
    delegation: { data: delegation, status },
    requiresAction:
      row.requiresAction || (!readOnly && status === "needs-input"),
  };
}

/** What hangs off a hand-off row on the wire, if anything. */
export function handoffNodeOf(
  row: ChainRow,
  readOnly: boolean,
  hasAnswer: boolean,
): HandoffNode {
  if (row.held?.state === "waiting") return null;
  if (row.state === "running" && row.output === undefined) return "loading";
  if (readOnly) return null;
  if (row.delegation?.status === "needs-input") return "question";
  return hasAnswer && row.delegation?.status === "running" ? "answered" : null;
}

/** The row's own words once the hand-off has a story to tell: the teammate
 *  asked something, or the user stopped them. */
export function describeHandoffRow(row: ChainRow, name: string): ChainRow {
  const info = row.delegation;
  if (!info) return row;
  if (info.status === "needs-input")
    return {
      ...row,
      text: `${name} asked a question`,
      state: "done",
      needsYou: true,
    };
  if (info.data.status === "cancelled" && !isTurnedDown(info.data))
    return {
      ...row,
      text: `You stopped the hand-off to ${name}`,
      state: "done",
      detail: undefined,
    };
  return row;
}
