import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { PageField } from "@/app/api/__generated__/models/pageField";
import type { ResolvedPage } from "@/app/api/__generated__/models/resolvedPage";

export type PageSection = ResolvedPage["sections"][number];

const ACTIVE_STATUSES: ReadonlySet<string> = new Set([
  AgentExecutionStatus.QUEUED,
  AgentExecutionStatus.RUNNING,
  AgentExecutionStatus.INCOMPLETE,
  AgentExecutionStatus.REVIEW,
]);

export function isRunActive(status: string | null | undefined) {
  return Boolean(status && ACTIVE_STATUSES.has(status));
}

/** Whether any result on the page is still running, so the page should poll. */
export function hasActiveResult(page: ResolvedPage | undefined) {
  return Boolean(
    page?.sections.some(
      (s) => s.kind === "latest_result" && isRunActive(s.run_status),
    ),
  );
}

export interface FieldGroup {
  name: string | null;
  fields: PageField[];
}

/** Consecutive fields that share a group show under one heading. */
export function groupFields(fields: PageField[]): FieldGroup[] {
  const groups: FieldGroup[] = [];
  for (const field of fields) {
    const name = field.group ?? null;
    const last = groups[groups.length - 1];
    if (last && last.name === name) last.fields.push(field);
    else groups.push({ name, fields: [field] });
  }
  return groups;
}

export function defaultValues(fields: PageField[]) {
  return Object.fromEntries(
    fields
      .filter((field) => field.default !== null && field.default !== undefined)
      .map((field) => [field.input, field.default]),
  );
}

const TITLES: Record<string, string> = {
  quick_runs: "Quick runs",
  latest_result: "Latest result",
  recent_runs: "Recent runs",
  run_history: "Run history",
  run_numbers: "Numbers",
  schedule: "Schedule",
  run_trend: "Trend",
  expert_status: "Status",
  expert_work: "Recent work",
  expert_learned: "What it has learned",
  expert_activity: "Activity",
  waiting: "Waiting on you",
};

/** The heading a section shows: its kind, and the workflow or expert it reads. */
export function sectionTitle(section: PageSection): string | null {
  if (section.kind === "text") return section.heading ?? null;
  if (section.kind === "cards" || section.kind === "notice") return null;
  if (section.kind === "run_form") return section.workflow_name;
  const base = TITLES[section.kind] ?? null;
  const source = sectionSource(section);
  return base && source ? `${base} · ${source}` : base;
}

function sectionSource(section: PageSection) {
  if ("workflow_name" in section) return section.workflow_name;
  if ("expert_name" in section) return section.expert_name;
  if (section.kind === "waiting") return section.scope_name ?? null;
  return null;
}

export function libraryRunHref(libraryAgentID: string, runID: string) {
  return `/library/agents/${libraryAgentID}?activeItem=${runID}`;
}

export function dollars(cents: number) {
  return `$${(cents / 100).toFixed(2)}`;
}
