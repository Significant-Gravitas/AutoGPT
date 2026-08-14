import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertPod } from "@/app/api/__generated__/models/expertPod";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { formatDistanceToNow } from "date-fns";

interface PodGroup {
  pod: ExpertPod;
  experts: Expert[];
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

export function workflowNeedsSetup(workflow: ExpertWorkflowRef) {
  return Boolean(workflow.schedule_cron) && !workflow.schedule_id;
}

export function getNeedsSetupCount(expert: Expert) {
  return expert.workflows.filter(workflowNeedsSetup).length;
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
