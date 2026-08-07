import type { Expert } from "@/app/api/__generated__/models/expert";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import {
  getExpertSchedules,
  getLastRunLabel,
  getNeedsSetupCount,
} from "@/app/(platform)/team/helpers";

export function getExpertStatusLine(
  expert: Expert,
  schedules: GraphExecutionJobInfo[],
): string {
  if (expert.schedules_paused_at) return "Paused";

  const needsSetup = getNeedsSetupCount(expert);
  if (needsSetup > 0) {
    return `${needsSetup} workflow${needsSetup === 1 ? "" : "s"} need setup`;
  }

  const lastRun = getLastRunLabel(expert);
  if (lastRun) return lastRun;

  const next = getExpertSchedules(expert, schedules)[0]?.next_run_time;
  if (next) {
    return `Next run ${new Date(next).toLocaleTimeString([], {
      hour: "numeric",
      minute: "2-digit",
    })}`;
  }

  return "Idle";
}
