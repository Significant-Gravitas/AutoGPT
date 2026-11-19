/** How often a new task runs, as the dialog offers it. Every rhythm is
 *  one cron in the person's own zone (the server's default). */
export type RepeatKind = "hourly" | "daily" | "weekdays" | "weekly";

export interface RepeatChoice {
  kind: RepeatKind;
  /** "HH:MM", for every rhythm but hourly. */
  time: string;
  /** 0 Sunday … 6 Saturday, for weekly. */
  day: number;
}

export const DEFAULT_TIME = "09:00";
export const DEFAULT_DAY = 1;

export const REPEAT_KINDS: { kind: RepeatKind; label: string }[] = [
  { kind: "hourly", label: "Every hour" },
  { kind: "daily", label: "Every day" },
  { kind: "weekdays", label: "Weekdays" },
  { kind: "weekly", label: "Every week" },
];

export const WEEK_DAYS = [
  "Sunday",
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
];

/** "HH:MM" as hour and minute; null for a time that does not read. */
function clock(time: string) {
  const match = /^(\d{1,2}):(\d{2})$/.exec(time.trim());
  if (!match) return null;
  const hour = Number(match[1]);
  const minute = Number(match[2]);
  if (hour > 23 || minute > 59) return null;
  return { hour, minute };
}

/** The cron for a rhythm; null while its time does not read. */
export function cronOf(choice: RepeatChoice) {
  if (choice.kind === "hourly") return "0 * * * *";
  const at = clock(choice.time);
  if (!at) return null;
  const head = `${at.minute} ${at.hour} * *`;
  switch (choice.kind) {
    case "daily":
      return `${head} *`;
    case "weekdays":
      return `${head} 1-5`;
    case "weekly":
      return `${head} ${choice.day}`;
  }
}

/** The rhythm in words: "Every day at 09:00", "Every Monday at 09:00". */
export function rhythmOf(choice: RepeatChoice) {
  const at = clock(choice.time) ? choice.time : DEFAULT_TIME;
  switch (choice.kind) {
    case "hourly":
      return "Every hour";
    case "daily":
      return `Every day at ${at}`;
    case "weekdays":
      return `Every weekday at ${at}`;
    case "weekly":
      return `Every ${WEEK_DAYS[choice.day] ?? WEEK_DAYS[DEFAULT_DAY]} at ${at}`;
  }
}
