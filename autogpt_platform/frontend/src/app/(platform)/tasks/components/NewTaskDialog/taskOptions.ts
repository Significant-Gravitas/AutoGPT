import type { TaskCreateRequest } from "@/app/api/__generated__/models/taskCreateRequest";
import { cronOf, type RepeatChoice } from "./repeat";
import { resolveStart, type StartChoice } from "./startChoice";

/** When and how a new task starts, and what done looks like. A task has
 *  a start time or a rhythm, never both. */
export interface TaskOptions {
  start: StartChoice | null;
  repeat: RepeatChoice | null;
  doneWhen: string;
  goal: string;
}

export const NO_OPTIONS: TaskOptions = {
  start: null,
  repeat: null,
  doneWhen: "",
  goal: "",
};

/** Anything set: the options row opens with it. */
export function hasOptions(options: TaskOptions) {
  return (
    isTimed(options) ||
    options.doneWhen.trim() !== "" ||
    options.goal.trim() !== ""
  );
}

/** Which part of the dialog a refusal is about. */
export type OptionsField = "start" | "repeat" | "form";

/** A task with a start time or a rhythm waits for it, so it is saved, not
 *  started. */
export function isTimed(options: TaskOptions) {
  return options.start !== null || options.repeat !== null;
}

/** The fields the options add to the create request, a quick start worked
 *  out at `now` (when it is sent); empty ones are left out. */
export function optionsRequest(
  options: TaskOptions,
  now: number,
): Partial<TaskCreateRequest> {
  const startAt = resolveStart(options.start, now);
  const cron = options.repeat ? cronOf(options.repeat) : null;
  const doneWhen = options.doneWhen.trim();
  const goal = options.goal.trim();
  return {
    ...(startAt ? { start_at: startAt } : {}),
    ...(cron ? { repeat: { crons: [cron] } } : {}),
    ...(doneWhen ? { done_when: doneWhen } : {}),
    ...(goal ? { goal } : {}),
  };
}

const VALUE_ERROR = /^Value error,\s*/;

/** The server's sentence, without the validator's prefix. */
export function refusalText(message: string) {
  return message.replace(VALUE_ERROR, "").trim();
}

/** Where a refusal belongs: the repeat (its cron), the start time (or the
 *  start-or-repeat rule), or the form as a whole. */
export function fieldOfRefusal(message: string): OptionsField {
  if (/start time/i.test(message)) return "start";
  if (/cron|repeat/i.test(message)) return "repeat";
  return "form";
}
