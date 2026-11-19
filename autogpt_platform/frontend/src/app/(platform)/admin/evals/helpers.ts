import type { EvalCase } from "@/app/api/__generated__/models/evalCase";
import type { EvalRunDetail } from "@/app/api/__generated__/models/evalRunDetail";
import type { EvalRunSummary } from "@/app/api/__generated__/models/evalRunSummary";

import type {
  EvalCase as ScreenCase,
  EvalChanges,
  EvalRunOption,
  EvalStat,
  EvalStatus,
  EvalVersion,
} from "./affect/evals";

const BEHAVIOURS: Record<string, string> = {
  reply: "Answers",
  start_task: "Starts a task",
  ask: "Asks a question",
  sign_in: "Shows a sign-in card",
  refuse: "Declines",
  delegate: "Hands it to an expert",
  task_update: "Updates a task",
  finish_task: "Finishes the task",
};

export function behaviourLabel(behaviour?: string | null): string {
  if (!behaviour) return "Not recorded";
  return BEHAVIOURS[behaviour] ?? behaviour.replaceAll("_", " ");
}

export function categoryLabel(name: string): string {
  const words = name.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function statusOf(evalCase: EvalCase): EvalStatus {
  return evalCase.verdict ?? "skipped";
}

export function toScreenCases(cases: EvalCase[]): ScreenCase[] {
  return cases.map((c) => ({
    id: c.case_id,
    category: c.category,
    input: c.title || c.request || "",
    expected: expectedText(c),
    output: outputText(c),
    score: c.score ?? null,
    status: statusOf(c),
    checks: checksText(c),
    search: searchText(c),
  }));
}

function expectedText(c: EvalCase): string {
  if (!c.expected) return "–";
  const must = c.expected.must?.[0];
  const behaviour = c.expected.behaviour
    ? behaviourLabel(c.expected.behaviour)
    : "";
  return [behaviour, must].filter(Boolean).join(" · ") || "–";
}

function outputText(c: EvalCase): string {
  if (c.skipped) return `Skipped: ${c.skipped}`;
  if (c.error) return `Broke: ${c.error}`;
  const reason = c.ratings?.[0]?.reasons?.[0];
  if (c.observed) {
    const did = behaviourLabel(c.observed);
    return reason && c.verdict !== "pass" ? `${did} · ${reason}` : did;
  }
  return reason ?? "Not saved for this run";
}

function checksText(c: EvalCase): string {
  const checks = c.checks ?? [];
  if (!checks.length) return "–";
  const failed = checks.filter((check) => !check.passed).length;
  if (c.detail === "report") return `${failed} failed`;
  return `${checks.length - failed}/${checks.length}`;
}

function searchText(c: EvalCase): string {
  const reasons = (c.ratings ?? []).flatMap((r) => r.reasons ?? []);
  return [c.case_id, c.title, c.request, c.world, ...reasons]
    .join(" ")
    .toLowerCase();
}

export function toStats(detail: EvalRunDetail): EvalStat[] {
  const { stats } = detail.run;
  const change = detail.comparison;
  const note = change ? `vs ${change.previous_label}` : "";
  const last: EvalStat = detail.seconds_per_turn
    ? {
        label: "Time per turn",
        value: String(detail.seconds_per_turn),
        unit: "s",
        delta: "",
        good: true,
        note: "",
      }
    : {
        label: "Partial",
        value: String(stats.partial),
        unit: ` of ${stats.cases}`,
        delta: "",
        good: true,
        note: "",
      };
  return [
    {
      label: "Pass rate",
      value: `${stats.pass_rate}%`,
      delta: change ? signed(change.pass_rate_delta) : "",
      good: !change || change.pass_rate_delta >= 0,
      note,
    },
    {
      label: "Avg score",
      value: stats.score.toFixed(1),
      unit: "/100",
      delta: change ? signed(change.score_delta) : "",
      good: !change || change.score_delta >= 0,
      note,
    },
    {
      label: "Failed",
      value: String(stats.failed),
      unit: ` of ${stats.cases}`,
      delta: change ? signed(change.failed_delta) : "",
      good: !change || change.failed_delta <= 0,
      note,
    },
    last,
  ];
}

function signed(value: number): string {
  const rounded = Math.round(value * 10) / 10;
  if (rounded === 0) return "±0";
  return `${rounded > 0 ? "+" : "−"}${Math.abs(rounded)}`;
}

export function toVersions(detail: EvalRunDetail): EvalVersion[] {
  return (detail.history ?? []).map((point) => ({
    id: point.id,
    label: point.label,
    rate: point.pass_rate,
    edited: `${point.pass_rate}% pass · score ${point.score}`,
  }));
}

export function toChanges(detail: EvalRunDetail): EvalChanges {
  const change = detail.comparison;
  const deltas = new Map(
    (change?.categories ?? []).map((c) => [c.name, c.pass_rate_delta ?? null]),
  );
  return {
    since: change ? `Since ${change.previous_label}` : null,
    categories: detail.categories.map((c) => ({
      name: c.name,
      label: categoryLabel(c.name),
      rate: c.stats.pass_rate,
      delta: deltas.get(c.name) ?? null,
      cases: c.stats.cases,
      failed: c.stats.failed,
    })),
    compared: Boolean(change?.cases_compared),
    worse: (change?.regressed ?? []).map((c) => ({
      id: c.case_id,
      title: c.title ?? "",
      before: c.before,
      after: c.after,
    })),
    better: (change?.improved ?? []).map((c) => ({
      id: c.case_id,
      title: c.title ?? "",
      before: c.before,
      after: c.after,
    })),
  };
}

export function toRunOptions(runs: EvalRunSummary[]): EvalRunOption[] {
  return runs.map((run) => ({
    id: run.id,
    label: run.label,
    description: [
      run.suite === "features" ? "Task features" : "Task eval",
      `${run.stats.pass_rate}% pass`,
      coverageText(run),
    ].join(" · "),
  }));
}

function coverageText(run: EvalRunSummary): string {
  if (run.case_coverage === "none") return "totals only";
  if (run.case_coverage === "not_passed") return "failures only";
  return `${run.stats.cases} cases`;
}

export function modelLabel(run: EvalRunSummary): string {
  if (run.suite === "features" && !run.otto_model) return "code checks only";
  return run.otto_model ?? "model not recorded";
}

export function runMeta(run: EvalRunSummary): string {
  const ran = formatRunDate(run.started_at);
  return ran ? `Ran ${ran}` : "Date not saved";
}

export function formatRunDate(value?: Date | string | null): string | null {
  if (!value) return null;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString("en-GB", {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}
