import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { formatDistanceToNow } from "date-fns";

/** Section headings sit outside the cards, so they need the cards' own content
 *  inset (1px border-box padding + p-4, matching AutopilotCard's p-5) to line
 *  up with the text inside them instead of with the card edge. */
export const SECTION_INSET_CLASS = "px-5";

/** Cancels an inherited `SECTION_INSET_CLASS`, for card grids that should keep
 *  running to the container edge inside an inset parent. */
export const SECTION_OUTSET_CLASS = "-mx-5";

/** The Button atom is pill-shaped by default; team actions sit in rows and read
 *  better squared off. */
export const ACTION_BUTTON_CLASS = "!rounded-xl";

/** The `outline` variant's zinc-700 border is too heavy at this size, so team
 *  actions soften it while keeping the hover lift. */
export const OUTLINE_ACTION_BUTTON_CLASS = `${ACTION_BUTTON_CLASS} !border-zinc-200 hover:!border-zinc-300`;

interface PodGroup {
  pod: ExpertPod;
  experts: Expert[];
}

export const TEAM_GRID_CLASS =
  "grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3";

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

export function getWeeklySpend(expert: Expert) {
  if (expert.weekly_budget == null || expert.weekly_budget <= 0) return null;
  return { spent: expert.weekly_spend ?? 0, budget: expert.weekly_budget };
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

export function getScheduleCountLabel(schedules: GraphExecutionJobInfo[]) {
  if (schedules.length === 0) return null;
  const count = `${schedules.length} ${schedules.length === 1 ? "schedule" : "schedules"}`;
  const next = schedules[0]?.next_run_time
    ? new Date(schedules[0].next_run_time)
    : null;
  if (!next || Number.isNaN(next.valueOf())) return count;
  return `${count} · next ${formatDistanceToNow(next, { addSuffix: true })}`;
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

interface AutopilotSummaryArgs {
  experts: Expert[];
  schedulesForExpert: (expert: Expert) => GraphExecutionJobInfo[];
}

/** The two things a daily user wants off the Autopilot card: what fires next
 *  across the whole team, and how much is stuck waiting on them. */
export function getAutopilotSummary({
  experts,
  schedulesForExpert,
}: AutopilotSummaryArgs) {
  const allSchedules = experts.flatMap(schedulesForExpert);
  const nextSchedule = allSchedules
    .filter((schedule) => Number.isFinite(nextRunMs(schedule)))
    .sort((a, b) => nextRunMs(a) - nextRunMs(b))[0];

  const pausedCount = experts.filter((expert) =>
    Boolean(expert.schedules_paused_at),
  ).length;
  const needsSetupCount = experts.reduce(
    (total, expert) =>
      total + getNeedsSetupCount(expert, schedulesForExpert(expert)),
    0,
  );

  return {
    nextRun: nextSchedule
      ? {
          name: nextSchedule.name,
          when: formatDistanceToNow(new Date(nextSchedule.next_run_time!), {
            addSuffix: true,
          }),
        }
      : null,
    attentionCount: pausedCount + needsSetupCount,
  };
}
