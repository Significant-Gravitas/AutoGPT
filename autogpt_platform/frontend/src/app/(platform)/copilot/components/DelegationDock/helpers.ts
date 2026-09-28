import type { UIDataTypes, UIMessage, UITools } from "ai";
import {
  type ChatDelegation,
  type LiveDelegationStatus,
  countDelegations,
  getDelegationSummary,
} from "../../delegations";
import { getLatestTaskList, isAllComplete } from "../TaskProgressBar/helpers";

/** The task progress bar owns the slot above the composer while the plan
 *  it shows is still open. */
export function hasRunningTaskList(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
): boolean {
  const todos = getLatestTaskList(messages);
  return !!todos && !isAllComplete(todos);
}

export interface DockLine {
  text: string;
  tone: "working" | "waiting" | "done" | "failed";
}

/** Folds each hand-off's live status (where a poll has one) over the frozen
 *  transcript status, then words the bar. A teammate who stopped on a
 *  question counts as waiting; a finished run drops out of the counts. */
export function getDockLine(
  delegations: ChatDelegation[],
  live: Record<string, LiveDelegationStatus>,
): DockLine | null {
  if (delegations.length === 0) return null;
  let needsInput = 0;
  const corrected = delegations.map((delegation) => {
    const status = live[delegation.toolCallId] ?? delegation.status;
    if (status === "needs-input") {
      needsInput += 1;
      return { ...delegation, status: "completed" as const };
    }
    if (status === "unknown")
      return { ...delegation, status: "running" as const };
    return { ...delegation, status };
  });
  const counts = countDelegations(corrected);
  const parts: string[] = [];
  if (counts.proposed > 0)
    parts.push(
      `${counts.proposed} hand-off${counts.proposed === 1 ? "" : "s"} waiting for your approval`,
    );
  if (needsInput > 0)
    parts.push(
      `${needsInput} expert${needsInput === 1 ? "" : "s"} need${needsInput === 1 ? "s" : ""} you`,
    );
  if (parts.length > 0) {
    const rest = getDelegationSummary({ ...counts, proposed: 0 });
    if (rest && counts.working + counts.queued > 0) parts.push(rest);
    return { text: parts.join(" · "), tone: "waiting" };
  }
  const summary = getDelegationSummary(counts);
  if (!summary) return null;
  if (counts.working + counts.queued > 0)
    return { text: summary, tone: "working" };
  return { text: summary, tone: "failed" };
}
