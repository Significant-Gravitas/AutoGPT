import { formatStartTime, QUICK_STARTS } from "../../schedule";

/** When a new task starts, as picked: a quick start, worked out when the
 *  task is sent rather than when it was picked; a date and time; or a date
 *  still to pick. No choice means now. */
export type StartChoice =
  | { kind: "quick"; label: string }
  | { kind: "picking" }
  | { kind: "at"; at: Date };

const HOUR = 60 * 60_000;
const MORNING_HOUR = 9;

/** The quick starts the dialog offers ("In 5 min" is the sidebar's). */
export const DIALOG_QUICK_STARTS = QUICK_STARTS.filter(
  (start) => start.label !== "In 5 min",
);

/** The time a quick start means if sent at `now`; null for the others. */
function quickAt(label: string, now: number) {
  return DIALOG_QUICK_STARTS.find((start) => start.label === label)?.at(now);
}

/** The start time to send, worked out at `now`; null to start now or
 *  while a date is still to pick. */
export function resolveStart(start: StartChoice | null, now: number) {
  if (!start || start.kind === "picking") return null;
  if (start.kind === "at") return start.at;
  return quickAt(start.label, now) ?? null;
}

function sameDay(a: Date, b: Date) {
  return a.toDateString() === b.toDateString();
}

/** The time a newly picked day starts at: the next round hour when it is
 *  today, 09:00 on a later day; never midnight. */
export function defaultStartOn(day: Date, now: number) {
  const at = new Date(day);
  if (sameDay(at, new Date(now))) {
    const next = new Date(now + HOUR);
    next.setMinutes(0, 0, 0);
    return next;
  }
  at.setHours(MORNING_HOUR, 0, 0, 0);
  return at;
}

/** A quick start's menu line, with the time it means now: "In 1 hour ·
 *  3:20 PM". */
export function quickLabel(label: string, now: number) {
  const at = quickAt(label, now);
  return at ? `${label} · ${formatStartTime(at, now)}` : label;
}

/** What the Start chip says. A picked date and time shows in its own
 *  field, so the chip does not repeat it. */
export function startChipLabel(start: StartChoice | null, now: number) {
  if (!start) return "Starts now";
  if (start.kind === "picking") return "Starts: pick a time";
  if (start.kind === "at") return "Starts on a set date";
  const label = quickLabel(start.label, now);
  return `Starts ${label.charAt(0).toLowerCase()}${label.slice(1)}`;
}
