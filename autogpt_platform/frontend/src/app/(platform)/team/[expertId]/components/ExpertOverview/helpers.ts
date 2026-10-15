import type { ExpertActivityDay } from "@/app/api/__generated__/models/expertActivityDay";
import { format, parseISO } from "date-fns";

export const OVERVIEW_WINDOW_DAYS = 30;

// Categorical slots 1 and 2 of the dataviz reference palette, validated as a
// pair on the light surface (normal-vision ΔE 33.6, CVD ΔE 24.7).
export const SERIES_COLORS = {
  sessions: "#2a78d6",
  runs: "#eb6834",
} as const;

function dayKey(day: ExpertActivityDay["day"]): string {
  return typeof day === "string"
    ? (day as string).slice(0, 10)
    : format(day, "yyyy-MM-dd");
}

export interface ChartDay {
  key: string;
  label: string;
  sessions: number;
  runs: number;
}

/** The last 30 days, oldest first, shaped for the activity chart. */
export function getChartDays(days: ExpertActivityDay[]): ChartDay[] {
  return days.slice(-OVERVIEW_WINDOW_DAYS).map((day) => {
    const key = dayKey(day.day);
    return {
      key,
      label: format(parseISO(key), "MMM d"),
      sessions: day.sessions,
      runs: day.runs,
    };
  });
}

function sumWindow(
  days: ExpertActivityDay[],
  field: "sessions" | "runs",
  windowsBack: number,
) {
  const end = days.length - OVERVIEW_WINDOW_DAYS * windowsBack;
  const start = Math.max(0, end - OVERVIEW_WINDOW_DAYS);
  return days
    .slice(start, Math.max(0, end))
    .reduce((total, day) => total + day[field], 0);
}

/** This window's total and how it compares with the window before it. */
export function getWindowStat(
  days: ExpertActivityDay[],
  field: "sessions" | "runs",
) {
  const current = sumWindow(days, field, 0);
  const previous = sumWindow(days, field, 1);
  return { current, delta: current - previous };
}

export function formatDelta(delta: number): string {
  if (delta === 0) return "Same as the 30 days before";
  return `${delta > 0 ? "+" : "−"}${Math.abs(delta)} vs the 30 days before`;
}

/** The bio's opening sentences, whole: the overview leads with these instead
 *  of clipping the paragraph mid-sentence. */
export function getBioLead(bio: string, sentences = 2): string {
  const parts = bio.trim().split(/(?<=[.!?])\s+/);
  return parts.slice(0, sentences).join(" ");
}
