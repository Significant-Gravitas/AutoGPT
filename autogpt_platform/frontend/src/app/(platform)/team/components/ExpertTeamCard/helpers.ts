import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { formatDistanceToNow } from "date-fns";

export function getScheduleSummary(expert: Expert) {
  const crons = expert.workflows
    .map((workflow) => workflow.schedule_cron)
    .filter((cron): cron is string => Boolean(cron));
  if (crons.length === 0) return null;
  const first = humanizeCronExpression(crons[0]);
  return crons.length > 1 ? `${first} +${crons.length - 1} more` : first;
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

export function workflowNeedsSetup(workflow: ExpertWorkflowRef) {
  return Boolean(workflow.schedule_cron) && !workflow.schedule_id;
}
