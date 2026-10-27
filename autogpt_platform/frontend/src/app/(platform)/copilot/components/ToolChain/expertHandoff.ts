import { asObject, str } from "./resultHelpers";

export interface HandoffTask {
  title: string;
  brief: string;
}

type HandoffStatus = "sent" | "queued" | "unconfirmed" | "failed";

export interface HandoffOutcome {
  title: string;
  status: HandoffStatus;
  subSessionId: string | null;
  link: string | null;
  /** Only for a failed hand-off: the backend says sending it again can help.
   *  A refusal (paused, out of budget, no free slot) is never retryable. */
  retry: boolean;
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
    subSessionId: str(handoff, "sub_session_id") ?? null,
    link: str(handoff, "link") ?? null,
    retry: handoff.retry === true,
  };
}

function statusOf(handoff: Record<string, unknown>): HandoffStatus {
  const status = str(handoff, "status");
  return status === "sent" || status === "queued" || status === "unconfirmed"
    ? status
    : "failed";
}

/** The status the confirm froze for a task that reached the thread, so the
 *  card can follow the thread live from there. An unconfirmed task may never
 *  have started, so an idle thread proves nothing about it. */
export function frozenStatusOf(outcome: HandoffOutcome): string | null {
  if (outcome.status === "sent") return "running";
  if (outcome.status === "queued") return "queued";
  return null;
}

export type HandoffState =
  | "failed"
  | "unconfirmed"
  | "waiting"
  | "working"
  | "stopped"
  | "unknown";

/** What the card says now: the thread's live status once the poll has read
 *  it, and the outcome frozen at hire time only until the first read. A poll
 *  that died ("unknown") claims nothing: the task was sent, and whether it
 *  is running is not known. */
export function handoffStateOf(
  outcome: HandoffOutcome,
  live: string | null,
): HandoffState {
  if (outcome.status === "failed") return "failed";
  if (outcome.status === "unconfirmed") return "unconfirmed";
  if (live === "unknown") return "unknown";
  if (live === "completed") return "stopped";
  if (live === "queued") return "waiting";
  if (live === "running") return "working";
  return outcome.status === "queued" ? "waiting" : "working";
}

/** Subject of the decision line for "hire and start". It reads as one
 *  phrase after the verb, so the gate's regex and the decision note both
 *  keep working: "hire James to take on the Printify order pipeline". */
export function handoffSubject(name: string, task: HandoffTask): string {
  return `${name} to take on ${task.title}`;
}

export function possessive(name: string): string {
  return `${name}'s`;
}
