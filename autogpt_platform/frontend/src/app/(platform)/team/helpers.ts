import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { formatDistanceToNow } from "date-fns";

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
  peerWorkflows: ExpertWorkflowRef[],
) {
  const sameGraphWorkflowCount = peerWorkflows.filter(
    (peer) => peer.graph_id === workflow.graph_id,
  ).length;
  return schedules.filter(
    (schedule) =>
      schedule.id === workflow.schedule_id ||
      (sameGraphWorkflowCount === 1 && schedule.graph_id === workflow.graph_id),
  );
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
  return expert.workflows.filter((workflow) => {
    const workflowSchedules = schedules
      ? getWorkflowSchedules(workflow, schedules, expert.workflows)
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
