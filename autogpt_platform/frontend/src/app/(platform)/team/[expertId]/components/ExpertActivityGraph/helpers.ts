import { ExpertActivityDay } from "@/app/api/__generated__/models/expertActivityDay";
import { format, subDays } from "date-fns";

export const DAYS_PER_WEEK = 7;
const DAYS_PER_MONTH = 30;
const THREE_MONTH_DAYS = 90;
const DAYS_PER_YEAR = 365;
const RECENT_WINDOW_DAYS = 7;
const LEVEL_COUNT = 4;

export type ActivityLevel = 0 | 1 | 2 | 3 | 4;

export interface ActivityCell {
  key: string;
  level: ActivityLevel;
  label: string | null;
}

/** The API declares `day` as a date, which the generated client types as
 *  `Date`, but a bare `YYYY-MM-DD` never passes the mutator's ISO date-time
 *  check so it reaches us as a string. Mock data does round-trip as a Date. */
function toDayKey(day: ExpertActivityDay["day"]) {
  return day instanceof Date
    ? day.toISOString().slice(0, 10)
    : String(day).slice(0, 10);
}

function parseDayKey(key: string) {
  const [year, month, day] = key.split("-").map(Number);
  return new Date(year, month - 1, day);
}

export function getYearActivityDays(days: ExpertActivityDay[]) {
  const lastDay =
    days.length > 0
      ? parseDayKey(toDayKey(days[days.length - 1].day))
      : new Date();
  const daysByKey = new Map(days.map((day) => [toDayKey(day.day), day]));

  return Array.from({ length: DAYS_PER_YEAR }, (_, index) => {
    const day = format(
      subDays(lastDay, DAYS_PER_YEAR - 1 - index),
      "yyyy-MM-dd",
    );
    return (
      daysByKey.get(day) ?? {
        day: day as unknown as Date,
        sessions: 0,
        runs: 0,
      }
    );
  });
}

export function getThreeMonthActivityDays(days: ExpertActivityDay[]) {
  return days.slice(-THREE_MONTH_DAYS);
}

export function getActivityTotal(day: ExpertActivityDay) {
  return day.sessions + day.runs;
}

export function getActivityLevel(total: number, max: number): ActivityLevel {
  if (total <= 0 || max <= 0) return 0;
  const level = Math.ceil((total / max) * LEVEL_COUNT);
  return Math.min(LEVEL_COUNT, Math.max(1, level)) as ActivityLevel;
}

function pluralize(count: number, noun: string) {
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}

export function describeActivityDay(day: ExpertActivityDay) {
  const when = format(parseDayKey(toDayKey(day.day)), "MMM d");
  const parts = [
    day.sessions > 0 ? pluralize(day.sessions, "session") : null,
    day.runs > 0 ? pluralize(day.runs, "run") : null,
  ].filter(Boolean);
  return parts.length > 0
    ? `${parts.join(", ")} on ${when}`
    : `No activity on ${when}`;
}

/** Cells in column-major order for a Sunday-first week grid: blank leading
 *  cells pad the first column so every column starts on a Sunday. */
export function getActivityCells(days: ExpertActivityDay[]): ActivityCell[] {
  if (days.length === 0) return [];
  const max = Math.max(...days.map(getActivityTotal));
  const leadingBlanks = parseDayKey(toDayKey(days[0].day)).getDay();
  const padding = Array.from({ length: leadingBlanks }, (_, index) => ({
    key: `pad-${index}`,
    level: 0 as ActivityLevel,
    label: null,
  }));
  return [
    ...padding,
    ...days.map((day) => ({
      key: toDayKey(day.day),
      level: getActivityLevel(getActivityTotal(day), max),
      label: describeActivityDay(day),
    })),
  ];
}

export function getActivitySummary(days: ExpertActivityDay[]) {
  const recent = days.slice(-RECENT_WINDOW_DAYS);
  const sessions = days.reduce((total, day) => total + day.sessions, 0);
  const runs = days.reduce((total, day) => total + day.runs, 0);
  const weeks = Math.max(1, Math.round(days.length / DAYS_PER_WEEK));
  return {
    isActive: recent.some((day) => getActivityTotal(day) > 0),
    rangeLabel:
      days.length >= DAYS_PER_YEAR
        ? "last year"
        : days.length >= THREE_MONTH_DAYS
          ? "last 3 months"
          : days.length >= DAYS_PER_MONTH
            ? "last month"
            : `last ${weeks} weeks`,
    totalsLabel: `${pluralize(sessions, "session")} · ${pluralize(runs, "run")}`,
  };
}

export function getActivityStreak(days: ExpertActivityDay[]) {
  let index = days.length - 1;
  if (index < 0) return 0;
  if (getActivityTotal(days[index]) === 0) index -= 1;

  let streak = 0;
  while (index >= 0 && getActivityTotal(days[index]) > 0) {
    streak += 1;
    index -= 1;
  }
  return streak;
}
