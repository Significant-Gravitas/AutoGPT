import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { findColorOption } from "@/app/(platform)/raise/components/ColorStep/helpers";
import { formatDistanceToNow } from "date-fns";

/** Section headings sit outside the cards, so they need the cards' own content
 *  inset (1px border-box padding + p-4, matching AutopilotCard's p-5) to line
 *  up with the text inside them instead of with the card edge. */
export const SECTION_INSET_CLASS = "px-4";

/** The Button atom is pill-shaped and tall by default; team actions match the
 *  home briefing's compact row buttons. */
export const ACTION_BUTTON_CLASS = "h-7 min-w-0 !rounded-md px-2.5 text-xs";

/** The `outline` variant's zinc-700 border is too heavy at this size, so team
 *  actions soften it while keeping the hover lift. */
export const OUTLINE_ACTION_BUTTON_CLASS = `${ACTION_BUTTON_CLASS} !border-zinc-200 hover:!border-zinc-300`;

interface PodGroup {
  pod: ExpertPod;
  experts: Expert[];
}

export const AUTOPILOT_ROLE = "Head of AI";

export const AUTOPILOT_BLURB =
  "Your built-in generalist. It answers questions, runs workflows, and delegates work across your hired experts.";

export const TEAM_GRID_CLASS =
  "grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3";

/** Mirrors `CreatePodRequest.name`'s `max_length` on the backend. */
export const POD_NAME_MAX_LENGTH = 100;

interface AssignToastArgs {
  podId: string | null;
  destinationName?: string;
}

/** `destinationName` is absent when the target pod is missing from the locally
 *  cached pod list, which the caller repairs by refetching pods. */
export function getAssignToastTitle({
  podId,
  destinationName,
}: AssignToastArgs) {
  if (podId === null) return "Removed from pod";
  return destinationName ? `Moved to ${destinationName}` : "Expert moved";
}

/** Split hired experts into their named pods (creation order, all pods kept
 *  so a just-created empty pod still shows) plus the ungrouped remainder. An
 *  expert whose `pod_id` points at a missing pod falls back to ungrouped. */
export function groupExpertsByPods(
  experts: Expert[],
  pods: ExpertPod[],
): { groups: PodGroup[]; ungrouped: Expert[] } {
  const membersByPod = new Map<string, Expert[]>(
    pods.map((pod) => [pod.id, []]),
  );
  const ungrouped: Expert[] = [];
  for (const expert of experts) {
    const members = expert.pod_id ? membersByPod.get(expert.pod_id) : undefined;
    if (members) members.push(expert);
    else ungrouped.push(expert);
  }
  return {
    groups: pods.map((pod) => ({
      pod,
      experts: membersByPod.get(pod.id) ?? [],
    })),
    ungrouped,
  };
}

export function getLastRunLabel(expert: Expert) {
  if (!expert.last_run_at) return null;
  const when = formatDistanceToNow(new Date(expert.last_run_at), {
    addSuffix: true,
  });
  if (expert.last_run_status === "COMPLETED")
    return `Last run succeeded ${when}`;
  if (expert.last_run_status === "FAILED") return `Last run failed ${when}`;
  return `Last run ${when}`;
}

/** The line under an expert's name on their card: their tagline, falling back
 *  to the opening of their identity when they have none. */
export function getExpertBlurb(expert: Expert) {
  if (expert.tagline?.trim()) return expert.tagline;
  const lines = expert.identity
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  return lines.slice(0, 2).join(" ") || null;
}

export function getWeeklySpend(expert: Expert) {
  if (expert.weekly_budget == null || expert.weekly_budget <= 0) return null;
  return { spent: expert.weekly_spend ?? 0, budget: expert.weekly_budget };
}

export type ExpertRosterStatus = "idle" | "working" | "needs-you";

export function getExpertRosterStatus(
  expert: Expert,
  needsSetupCount: number,
): ExpertRosterStatus {
  const runStatus = expert.last_run_status?.toUpperCase();

  if (runStatus === "RUNNING" || runStatus === "QUEUED") return "working";
  if (
    expert.schedules_paused_at ||
    needsSetupCount > 0 ||
    runStatus === "FAILED" ||
    runStatus === "TERMINATED" ||
    runStatus === "REVIEW"
  ) {
    return "needs-you";
  }
  return "idle";
}

export function getWorkflowSchedules(
  workflow: ExpertWorkflowRef,
  schedules: GraphExecutionJobInfo[],
  graphWorkflowCounts: ReadonlyMap<ExpertWorkflowRef["graph_id"], number>,
) {
  const sameGraphWorkflowCount =
    graphWorkflowCounts.get(workflow.graph_id) ?? 0;
  return schedules.filter(
    (schedule) =>
      schedule.id === workflow.schedule_id ||
      (sameGraphWorkflowCount === 1 && schedule.graph_id === workflow.graph_id),
  );
}

export function getGraphWorkflowCounts(workflows: ExpertWorkflowRef[]) {
  const counts = new Map<ExpertWorkflowRef["graph_id"], number>();
  for (const workflow of workflows) {
    counts.set(workflow.graph_id, (counts.get(workflow.graph_id) ?? 0) + 1);
  }
  return counts;
}

export function workflowNeedsSetup(
  workflow: ExpertWorkflowRef,
  schedules?: GraphExecutionJobInfo[],
) {
  const hasSchedule = schedules
    ? schedules.length > 0
    : Boolean(workflow.schedule_id);
  return Boolean(workflow.schedule_cron) && !hasSchedule;
}

export function getNeedsSetupCount(
  expert: Expert,
  schedules?: GraphExecutionJobInfo[],
) {
  const graphWorkflowCounts = getGraphWorkflowCounts(expert.workflows);
  return expert.workflows.filter((workflow) => {
    const workflowSchedules = schedules
      ? getWorkflowSchedules(workflow, schedules, graphWorkflowCounts)
      : undefined;
    return workflowNeedsSetup(workflow, workflowSchedules);
  }).length;
}

/** Schedules belonging to an expert, soonest-firing first. Matches on the
 *  schedule's own expert_id, falling back to the workflow-ref join for
 *  schedules created before expert_id was stamped on them. */
export function getExpertSchedules(
  expert: Expert,
  schedules: GraphExecutionJobInfo[],
) {
  const workflowScheduleIds = new Set(
    expert.workflows
      .map((workflow) => workflow.schedule_id)
      .filter((id): id is string => Boolean(id)),
  );
  return schedules
    .filter(
      (schedule) =>
        schedule.expert_id === expert.id ||
        workflowScheduleIds.has(schedule.id),
    )
    .sort((a, b) => nextRunMs(a) - nextRunMs(b));
}

function nextRunMs(schedule: GraphExecutionJobInfo) {
  if (!schedule.next_run_time) return Number.POSITIVE_INFINITY;
  const t = new Date(schedule.next_run_time).valueOf();
  return Number.isNaN(t) ? Number.POSITIVE_INFINITY : t;
}

export type TeamFilter = "all" | "scheduled" | "needs-setup" | "paused";

interface FilterExpertsArgs {
  experts: Expert[];
  query: string;
  filter: TeamFilter;
  schedulesForExpert: (expert: Expert) => GraphExecutionJobInfo[];
}

/** Narrows the roster by free-text (name or role) and by the toolbar's filter.
 *  Both are applied together, so an empty result can mean either. */
export function filterExperts({
  experts,
  query,
  filter,
  schedulesForExpert,
}: FilterExpertsArgs) {
  const needle = query.trim().toLowerCase();
  return experts.filter((expert) => {
    const haystack = `${expert.name} ${expert.role}`.toLowerCase();
    if (needle && !haystack.includes(needle)) return false;
    if (filter === "scheduled") return schedulesForExpert(expert).length > 0;
    if (filter === "needs-setup")
      return getNeedsSetupCount(expert, schedulesForExpert(expert)) > 0;
    if (filter === "paused") return Boolean(expert.schedules_paused_at);
    return true;
  });
}

/** Experts on the roster: hired (not marketplace templates) and not fired. */
export function getHiredExperts(experts: Expert[]) {
  return experts.filter((expert) => !expert.is_template && !expert.is_archived);
}

/** Every skill on the team, de-duplicated and alphabetised. */
export function getTeamSkills(experts: Expert[]) {
  return Array.from(new Set(experts.flatMap((expert) => expert.skills))).sort(
    (a, b) => a.localeCompare(b),
  );
}

/** Every expert's schedules as one list, soonest-firing first. A schedule
 *  reachable from two experts (shared graph) is listed once. */
export function getTeamSchedules(
  experts: Expert[],
  schedules: GraphExecutionJobInfo[],
) {
  const byId = new Map(
    experts
      .flatMap((expert) => getExpertSchedules(expert, schedules))
      .map((schedule) => [schedule.id, schedule]),
  );
  return Array.from(byId.values()).sort((a, b) => nextRunMs(a) - nextRunMs(b));
}

interface AutopilotSummaryArgs {
  experts: Expert[];
  schedulesForExpert: (expert: Expert) => GraphExecutionJobInfo[];
}

/** Autopilot works across the whole team, so its card counts the team's
 *  totals. Skills are de-duplicated — two experts who can both write copy is
 *  one skill on the team, not two. */
export function getAutopilotSummary({
  experts,
  schedulesForExpert,
}: AutopilotSummaryArgs) {
  return {
    skillCount: getTeamSkills(experts).length,
    scheduleCount: new Set(
      experts.flatMap(schedulesForExpert).map((schedule) => schedule.id),
    ).size,
    workflowCount: experts.reduce(
      (total, expert) => total + expert.workflows.length,
      0,
    ),
  };
}

/** Covers live at `public/experts/covers/<color token>.jpg`, one per raise-flow
 *  accent, plus `default.jpg` for experts without a colour (marketplace
 *  templates). */
export function getExpertCoverSrc(color: string | null | undefined) {
  const token = findColorOption(color ?? null)?.id ?? "default";
  return `/experts/covers/${token}.jpg`;
}

export const WORKFLOW_FILTERS = [
  { value: "all", label: "All workflows" },
  { value: "scheduled", label: "Scheduled" },
  { value: "manual", label: "Manual" },
  { value: "needs-setup", label: "Needs setup" },
] as const;

export type WorkflowFilter = (typeof WORKFLOW_FILTERS)[number]["value"];

export function filterExpertWorkflows(
  workflows: ExpertWorkflowRef[],
  query: string,
  filter: WorkflowFilter,
) {
  const needle = query.trim().toLowerCase();
  return workflows.filter((workflow) => {
    if (needle) {
      const haystack =
        `${workflow.name ?? ""} ${workflow.description ?? ""}`.toLowerCase();
      if (!haystack.includes(needle)) return false;
    }
    if (filter === "all") return true;
    if (filter === "needs-setup") return workflowNeedsSetup(workflow);
    if (filter === "scheduled")
      return Boolean(workflow.schedule_cron) && !workflowNeedsSetup(workflow);
    return !workflow.schedule_cron;
  });
}

export const SCHEDULE_FILTERS = [
  { value: "all", label: "All schedules" },
  { value: "today", label: "Next 24 hours" },
  { value: "week", label: "Next 7 days" },
  { value: "later", label: "Later" },
] as const;

export type ScheduleFilter = (typeof SCHEDULE_FILTERS)[number]["value"];

const DAY_MS = 24 * 60 * 60 * 1000;

export function filterExpertSchedules(
  schedules: GraphExecutionJobInfo[],
  query: string,
  filter: ScheduleFilter,
  now: Date = new Date(),
) {
  const needle = query.trim().toLowerCase();
  return schedules.filter((schedule) => {
    if (needle) {
      const haystack =
        `${schedule.name} ${schedule.agent_name ?? ""}`.toLowerCase();
      if (!haystack.includes(needle)) return false;
    }
    if (filter === "all") return true;
    const untilNext = nextRunMs(schedule) - now.getTime();
    if (untilNext < 0) return false;
    if (filter === "today") return untilNext <= DAY_MS;
    if (filter === "week") return untilNext <= 7 * DAY_MS;
    return untilNext > 7 * DAY_MS;
  });
}
