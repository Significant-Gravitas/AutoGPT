import { asObject, str } from "./resultHelpers";

export interface HandoffTask {
  title: string;
  brief: string;
}

type HandoffStatus = "started" | "queued" | "unconfirmed" | "failed";

export interface HandoffOutcome {
  title: string;
  status: HandoffStatus;
  link: string | null;
}

/** The task a hire was proposed for, when the proposal offers the hand-off. */
export function handoffTaskOf(
  output: Record<string, unknown>,
): HandoffTask | null {
  const handoff = asObject(output.handoff);
  if (!handoff || !str(output, "handoff_confirmation_id")) return null;
  const title = str(handoff, "title");
  const brief = str(handoff, "brief");
  return title && brief ? { title, brief } : null;
}

/** Whether the task a hire was approved with reached the new expert. */
export function handoffOutcomeOf(
  output: Record<string, unknown>,
): HandoffOutcome | null {
  const handoff = asObject(output.handoff);
  const title = handoff && str(handoff, "title");
  if (!handoff || !title) return null;
  return {
    title,
    status: statusOf(handoff),
    link: str(handoff, "link") ?? null,
  };
}

function statusOf(handoff: Record<string, unknown>): HandoffStatus {
  const status = str(handoff, "status");
  return status === "started" || status === "queued" || status === "unconfirmed"
    ? status
    : "failed";
}

/** Subject of the decision line for "hire and hand off". It reads as one
 *  phrase after the verb, so the gate's regex and the decision note both
 *  keep working: "hire James to take on the Printify order pipeline". */
export function handoffSubject(name: string, task: HandoffTask): string {
  return `${name} to take on ${task.title}`;
}

export function possessive(name: string): string {
  return `${name}'s`;
}
