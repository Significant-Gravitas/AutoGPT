import type { CopilotTurnJobInfo } from "@/app/api/__generated__/models/copilotTurnJobInfo";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { DEFAULT_AUTOPILOT_MODE } from "@/app/(platform)/copilot/autopilotModeStore";
import { getModeOption } from "@/app/(platform)/copilot/components/ChatInput/components/AutopilotModeSelector/helpers";
import { getWeeklySpend } from "@/app/(platform)/team/helpers";
import { creditsToUsdLabel } from "@/lib/credits";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { formatTimeAgo } from "@/lib/utils/time";
import { getExpertRoleLabel } from "@/services/experts/expert-role-label";
import {
  PlayCircleIcon,
  SecurityCheckIcon,
  Wallet01Icon,
} from "@hugeicons/core-free-icons";
import { format, formatDistanceToNowStrict } from "date-fns";

export interface PanelSchedule {
  id: string;
  title: string;
  cadence: string;
  nextRun: string | null;
  nextRunMs: number;
}

export function toPanelSchedules(
  graphs: GraphExecutionJobInfo[],
  followups: CopilotTurnJobInfo[],
): PanelSchedule[] {
  const rows = [
    ...graphs.map((schedule) => ({
      id: schedule.id,
      title: schedule.name || schedule.agent_name || "Workflow schedule",
      cadence: getCadence(schedule.cron),
      next: schedule.next_run_time,
    })),
    ...followups.map((followup) => ({
      id: followup.id,
      title: getFollowupTitle(followup),
      cadence: followup.cron
        ? getCadence(followup.cron)
        : getOneOffLabel(followup.run_at ?? followup.next_run_time),
      next: followup.next_run_time,
    })),
  ];
  return rows
    .map(({ next, ...row }) => ({
      ...row,
      nextRun: getNextRunLabel(next),
      nextRunMs: toMs(next),
    }))
    .sort((a, b) => a.nextRunMs - b.nextRunMs);
}

function getFollowupTitle(followup: CopilotTurnJobInfo) {
  const message = followup.message.trim().split("\n")[0];
  return followup.name || message || "Follow-up";
}

function getCadence(cron: string) {
  try {
    return humanizeCronExpression(cron);
  } catch {
    return cron;
  }
}

function getOneOffLabel(when: Date | string) {
  const date = new Date(when);
  if (Number.isNaN(date.valueOf())) return "Once";
  return `Once, ${format(date, "MMM d, h:mm a")}`;
}

function getNextRunLabel(when: string | null | undefined) {
  if (!when) return null;
  const next = new Date(when);
  if (Number.isNaN(next.valueOf())) return null;
  return formatDistanceToNowStrict(next, { addSuffix: true });
}

function toMs(when: string | null | undefined) {
  if (!when) return Number.POSITIVE_INFINITY;
  const ms = new Date(when).valueOf();
  return Number.isNaN(ms) ? Number.POSITIVE_INFINITY : ms;
}

export function getExpertRole(expert: Expert) {
  return expert.job_title || getExpertRoleLabel(expert.role) || "Expert";
}

export function getWorkInfo(expert: Expert) {
  const spend = getWeeklySpend(expert);
  return [
    {
      icon: SecurityCheckIcon,
      label: "Approval",
      value: getModeOption(expert.autopilot_mode ?? DEFAULT_AUTOPILOT_MODE)
        .label,
    },
    {
      icon: Wallet01Icon,
      label: "Weekly budget",
      value: spend
        ? `${creditsToUsdLabel(spend.spent)} of ${creditsToUsdLabel(spend.budget)}`
        : "No limit",
    },
    {
      icon: PlayCircleIcon,
      label: "Last run",
      value: expert.last_run_at
        ? formatTimeAgo(expert.last_run_at.toString())
        : "Not yet",
    },
  ];
}
