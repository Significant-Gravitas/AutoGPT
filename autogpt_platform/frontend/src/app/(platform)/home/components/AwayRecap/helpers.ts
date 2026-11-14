import type { HomeBriefingOutcome } from "@/app/api/__generated__/models/homeBriefingOutcome";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import type { Tone } from "@/components/styles/tones";
import {
  AlertCircleIcon,
  CheckmarkCircle02Icon,
  Notification02Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

export type RecapKind = "needs" | "failed" | "done";

export interface RecapItem {
  id: string;
  kind: RecapKind;
  /** When it happened, in ms; null when the backend gave no time. */
  at: number | null;
  title: string;
  detail: string;
  href: string | null;
  actionLabel: string;
}

interface Lane {
  kind: RecapKind;
  label: string;
  /** The recap's title while one of this lane's items is open. */
  heading: string;
  empty: string;
  icon: IconSvgElement;
  tone: Tone;
  /** The dot on the chart. */
  dot: string;
}

export const LANES: Lane[] = [
  {
    kind: "needs",
    label: "Needs you",
    heading: "Needs your input",
    empty: "Nothing needs you",
    icon: Notification02Icon,
    tone: "waiting",
    dot: "bg-status-waiting",
  },
  {
    kind: "failed",
    label: "Failed",
    heading: "Failed",
    empty: "Nothing failed",
    icon: AlertCircleIcon,
    tone: "error",
    dot: "bg-error",
  },
  {
    kind: "done",
    label: "Done",
    heading: "Done",
    empty: "Nothing finished yet",
    icon: CheckmarkCircle02Icon,
    tone: "ok",
    dot: "bg-status-ok",
  },
];

export const CARDS_PER_LANE = 3;

function toMs(value: Date | string | null | undefined) {
  if (!value) return null;
  const ms = new Date(value).getTime();
  return Number.isNaN(ms) ? null : ms;
}

function runsOf(dashboard: HomeDashboardResponse): HomeBriefingOutcome[] {
  const grouped = (dashboard.recent_work?.groups ?? []).flatMap(
    (group) => group.runs ?? [],
  );
  const runs = grouped.length > 0 ? grouped : dashboard.briefing.outcomes;
  const seen = new Set<string>();
  return runs.filter((run) => {
    if (seen.has(run.id)) return false;
    seen.add(run.id);
    return true;
  });
}

/** What needs the user, what failed and what finished, newest first. */
export function recapItemsOf(dashboard: HomeDashboardResponse): RecapItem[] {
  const needs: RecapItem[] = dashboard.attention.map((item) => ({
    id: `needs-${item.id}`,
    kind: "needs",
    at: toMs(item.created_at),
    title: item.title,
    detail: item.description,
    href: item.primary_action.href,
    actionLabel: item.primary_action.label,
  }));
  const runs: RecapItem[] = runsOf(dashboard).map((run) => ({
    id: `run-${run.id}`,
    kind: run.status === "failed" ? "failed" : "done",
    at: toMs(run.occurred_at),
    title: run.title,
    detail:
      run.status === "failed" && run.summary
        ? run.summary
        : (run.expert?.name ?? run.agent_name),
    href: run.link ?? null,
    actionLabel: "Open",
  }));
  return [...needs, ...runs].sort((a, b) => (b.at ?? 0) - (a.at ?? 0));
}

/** The stretch of time the chart covers. */
export function recapWindow(
  dashboard: HomeDashboardResponse,
  items: RecapItem[],
) {
  const end = toMs(dashboard.generated_at) ?? Date.now();
  const times = items.flatMap((item) => (item.at === null ? [] : [item.at]));
  const declared =
    toMs(dashboard.recent_work?.window_started_at) ??
    toMs(dashboard.briefing.window_started_at);
  const start = Math.min(declared ?? end, ...times, end - 60_000);
  return { start, end };
}

/** Where an item sits along the chart, 0 to 1; null when it has no time. */
export function positionOf(item: RecapItem, start: number, end: number) {
  if (item.at === null) return null;
  const t = (item.at - start) / Math.max(1, end - start);
  // Kept off the very edges so its dot is never clipped.
  return Math.min(0.985, Math.max(0.015, t));
}

export const CHART_WIDTH = 600;
export const CHART_HEIGHT = 96;
const HEADROOM = 16;
const FLOOR = 3;
const STEPS = 72;
const REACH = 0.045;
const BASELINE = 0.05;

// One soft hill per event, added together: busy stretches rise, quiet ones
// rest on the floor.
function activityAt(x: number, points: number[]) {
  return points.reduce((sum, t) => {
    const distance = Math.abs(x - t) / REACH;
    return distance >= 1 ? sum : sum + (1 + Math.cos(Math.PI * distance)) / 2;
  }, BASELINE);
}

export function activityChart(points: number[]) {
  const values = Array.from({ length: STEPS + 1 }, (_, i) =>
    activityAt(i / STEPS, points),
  );
  const peak = Math.max(...values, 0.4);
  function yOf(value: number) {
    const usable = CHART_HEIGHT - HEADROOM - FLOOR;
    return CHART_HEIGHT - FLOOR - (value / peak) * usable;
  }
  const coords = values.map((value, i) => ({
    x: (i / STEPS) * CHART_WIDTH,
    y: yOf(value),
  }));
  // Smoothed by curving through the midpoint of each pair of samples.
  const segments = coords.slice(1, -1).map((point, i) => {
    const next = coords[i + 2];
    return `Q ${point.x} ${point.y} ${(point.x + next.x) / 2} ${(point.y + next.y) / 2}`;
  });
  const last = coords[coords.length - 1];
  const line = `M ${coords[0].x} ${coords[0].y} ${segments.join(" ")} L ${last.x} ${last.y}`;
  return {
    line,
    area: `${line} L ${CHART_WIDTH} ${CHART_HEIGHT} L 0 ${CHART_HEIGHT} Z`,
    /** How far down the chart the line is at `t`, 0 (top) to 1 (bottom). */
    heightAt: (t: number) => yOf(activityAt(t, points)) / CHART_HEIGHT,
  };
}
