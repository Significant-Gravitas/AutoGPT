import type { UIDataTypes, UIMessage, UITools } from "ai";
import {
  type ChatDelegation,
  type LiveDelegationStatus,
  countDelegations,
  getDelegationSummary,
} from "../../delegations";
import { delegationName } from "../../delegationViews";
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

function plural(count: number, noun: string) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

/** Called before its tool returns: the teammate's run has no id yet. */
function isHandingOff(delegation: ChatDelegation): boolean {
  return delegation.status === "running" && !delegation.subSessionId;
}

function handingOffText(handingOff: ChatDelegation[], names: NameOf) {
  return handingOff.length === 1
    ? `Handing off to ${names(handingOff[0])}…`
    : `Handing off to ${handingOff.length} experts…`;
}

type NameOf = (delegation: ChatDelegation) => string;

/** Folds each hand-off's live status (where a poll has one) over the frozen
 *  transcript status, then words the bar. A teammate who stopped on a
 *  question counts as waiting; a finished run drops out of the counts. */
export function getDockLine(
  delegations: ChatDelegation[],
  live: Record<string, LiveDelegationStatus>,
  names: NameOf = (delegation) => delegationName(delegation),
): DockLine | null {
  if (delegations.length === 0) return null;
  const corrected = delegations.map((delegation) => {
    const status = live[delegation.toolCallId] ?? delegation.status;
    if (status === "unknown")
      return { ...delegation, status: "running" as const };
    return { ...delegation, status };
  });
  const handingOff = corrected.filter(isHandingOff);
  const counts = countDelegations(
    corrected.filter((delegation) => !isHandingOff(delegation)),
  );
  const parts: string[] = [];
  if (counts.proposed > 0)
    parts.push(
      `${plural(counts.proposed, "hand-off")} waiting for your approval`,
    );
  if (counts.needsInput > 0)
    parts.push(
      `${plural(counts.needsInput, "expert")} need${counts.needsInput === 1 ? "s" : ""} you`,
    );
  if (parts.length > 0) {
    const rest = getDelegationSummary({ ...counts, proposed: 0 });
    if (rest && counts.working + counts.queued > 0) parts.push(rest);
    return { text: parts.join(" · "), tone: "waiting" };
  }
  const summary = getDelegationSummary(counts);
  if (handingOff.length > 0) {
    const inFlight = counts.working + counts.queued > 0 ? summary : null;
    return {
      text: [handingOffText(handingOff, names), inFlight]
        .filter(Boolean)
        .join(" · "),
      tone: "working",
    };
  }
  if (!summary) return null;
  if (counts.working + counts.queued > 0)
    return { text: summary, tone: "working" };
  return { text: summary, tone: "failed" };
}
