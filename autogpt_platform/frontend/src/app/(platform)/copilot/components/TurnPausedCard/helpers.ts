import type { UIMessage } from "ai";

import { TURN_PAUSED_PART_TYPE, type TurnPause } from "../../turnPause";

/**
 * The pause the chat is sitting on: the last thing in it, with nothing drawn
 * after it. Once anything follows (a "Keep going" note, a resumed turn, the
 * user's next message) the pause is history and its card stops offering
 * ways back in. A card knows it is this pause by comparing its `data` to it
 * by reference, so the chat's message transforms must pass that object on.
 */
export function getCurrentPause(messages: readonly UIMessage[]): unknown {
  const last = messages[messages.length - 1];
  if (last?.role !== "assistant") return null;
  for (let i = last.parts.length - 1; i >= 0; i--) {
    const part = last.parts[i];
    if (part.type === TURN_PAUSED_PART_TYPE) {
      return (part as { data?: unknown }).data ?? null;
    }
    if (!drawsNothing(part)) return null;
  }
  return null;
}

function drawsNothing(part: UIMessage["parts"][number]): boolean {
  if (part.type === "step-start" || part.type.startsWith("data-")) return true;
  if (part.type === "text" || part.type === "reasoning") {
    return !part.text.trim();
  }
  return false;
}

type BudgetPause = TurnPause & { reason: "daily" | "weekly" };

export function isBudgetPause(pause: TurnPause): pause is BudgetPause {
  return pause.reason === "daily" || pause.reason === "weekly";
}

/** A budget pause with no reset time: the account has no plan, so its
 *  allowance never refills on its own. */
export function neverResets(pause: TurnPause): boolean {
  return isBudgetPause(pause) && pause.resets_at === null;
}

/** The window the pause waited on has refilled since. */
export function hasReset(pause: TurnPause, now = new Date()): boolean {
  if (!isBudgetPause(pause) || !pause.resets_at) return false;
  return Date.parse(pause.resets_at) <= now.getTime();
}

const BUDGET: Record<"daily" | "weekly", string> = {
  daily: "today's budget",
  weekly: "this week's budget",
};

export function pauseTitle(pause: TurnPause): string {
  const lead = pause.started ? "Paused" : "Not started";
  // No plan behind it, so no window that will refill to name.
  if (neverResets(pause)) return `${lead} · out of budget`;
  if (isBudgetPause(pause)) return `${lead} · out of ${BUDGET[pause.reason]}`;
  if (pause.reason === "trial") return `${lead} · trial allowance used`;
  if (!pause.started) return lead;
  return pause.reason === "task"
    ? "Paused · this task reached its size limit"
    : "Paused · checking in after a long run";
}

interface BodyArgs {
  pause: TurnPause;
  name: string;
  /** The budget refilled after the pause, so there is nothing to wait for. */
  budgetBack?: boolean;
  now?: Date;
}

export function pauseBody({
  pause,
  name,
  budgetBack = false,
  now = new Date(),
}: BodyArgs): string {
  if (pause.reason === "steps" && pause.started) return stepsBody(pause, name);
  if (!pause.started) return notStartedBody(pause, name, budgetBack, now);
  const saved = pause.checkpoint_saved
    ? `${name} saved its progress above.`
    : `${name} was stopped before it could summarize its progress.`;
  if (!isBudgetPause(pause)) return saved;
  if (budgetBack) return `${saved} Your budget has reset.`;
  const reset = resetPhrase(pause.resets_at, now);
  return reset ? `${saved} Your budget resets ${reset}.` : saved;
}

function stepsBody(pause: TurnPause, name: string): string {
  const done =
    pause.steps !== null
      ? `has taken ${pause.steps.toLocaleString()} steps on this`
      : "has been at this a while";
  return pause.checkpoint_saved
    ? `${name} ${done} and summarized its progress above.`
    : `${name} ${done} and was stopped before it could summarize its progress.`;
}

function notStartedBody(
  pause: TurnPause,
  name: string,
  budgetBack: boolean,
  now: Date,
): string {
  if (pause.reason === "trial") {
    return `${name} didn't start this because you're out of your trial allowance.`;
  }
  if (!isBudgetPause(pause)) return `${name} didn't start this.`;
  if (neverResets(pause)) {
    return `${name} didn't start this because you're out of budget.`;
  }
  const budget = BUDGET[pause.reason];
  if (budgetBack) {
    return `${name} didn't start this because you were out of ${budget}. It has reset since.`;
  }
  const reset = resetPhrase(pause.resets_at, now);
  const lead = `${name} didn't start this because you're out of ${budget}.`;
  return reset ? `${lead} It resets ${reset}.` : lead;
}

/** "at 1:00 AM, in 4 h 12 min", or nothing when the time is unknown. */
function resetPhrase(resetsAt: string | null, now: Date): string | null {
  if (!resetsAt) return null;
  const date = new Date(resetsAt);
  if (!Number.isFinite(date.getTime())) return null;
  return `${resetAt(date, now)}, ${resetIn(date, now)}`;
}

/** "Resume at 1:00 AM", or a plain "Resume at reset" when the time is unknown. */
export function resumeLabel(resetsAt: string | null, now = new Date()) {
  const date = resetsAt ? new Date(resetsAt) : null;
  return date && Number.isFinite(date.getTime())
    ? `Resume ${resetAt(date, now)}`
    : "Resume at reset";
}

const DAY_MS = 24 * 60 * 60 * 1000;

/** "at 1:00 AM" within a day, "Monday at 1:00 AM" within a week. */
export function resetAt(date: Date, now = new Date()): string {
  const time = date.toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
  const ms = date.getTime() - now.getTime();
  if (ms < DAY_MS) return `at ${time}`;
  const day =
    ms < 7 * DAY_MS
      ? date.toLocaleDateString(undefined, { weekday: "long" })
      : date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  return `${day} at ${time}`;
}

/** "in 4 h 12 min", "in 45 min", "in 2 days". */
export function resetIn(date: Date, now = new Date()): string {
  const ms = date.getTime() - now.getTime();
  const minutes = Math.max(1, Math.ceil(ms / 60_000));
  if (minutes < 60) return `in ${minutes} min`;
  // resetAt's cut-off: a time it names without the day never reads "in 1 day".
  if (ms < DAY_MS) {
    const hours = Math.floor(minutes / 60);
    const rest = minutes % 60;
    return rest ? `in ${hours} h ${rest} min` : `in ${hours} h`;
  }
  const days = Math.round(minutes / (24 * 60));
  return `in ${days} ${days === 1 ? "day" : "days"}`;
}
