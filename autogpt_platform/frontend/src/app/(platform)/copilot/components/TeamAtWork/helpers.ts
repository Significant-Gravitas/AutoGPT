import type { ExpertFirstJob } from "@/app/api/__generated__/models/expertFirstJob";
import type { ExpertFirstJobStatus } from "@/app/api/__generated__/models/expertFirstJobStatus";
import { serviceLabelFromIcon } from "@/components/contextual/IntegrationsPanel/helpers";
import { trackFunnel } from "@/services/experts/experts-analytics";

const STATUS_CONFIG: Record<
  ExpertFirstJobStatus,
  { label: string; dot: string; pulse: boolean }
> = {
  installing: { label: "Installing", dot: "bg-zinc-400", pulse: true },
  needs_setup: { label: "Needs setup", dot: "bg-warning", pulse: false },
  needs_you: { label: "Needs you", dot: "bg-warning", pulse: false },
  working: { label: "Working", dot: "bg-primary", pulse: true },
  ready: { label: "Ready for review", dot: "bg-emerald-500", pulse: false },
  open: { label: "Start a chat", dot: "bg-zinc-300", pulse: false },
};

const CHAT_STATUSES: ExpertFirstJobStatus[] = ["working", "ready", "needs_you"];
const MOVING_STATUSES: ExpertFirstJobStatus[] = ["installing", "working"];

export function getFirstJobHref(item: ExpertFirstJob) {
  if (item.status === "open") {
    return `/copilot?expertId=${item.expert_id}&kickoff=1`;
  }
  if (CHAT_STATUSES.includes(item.status) && item.session_id) {
    return `/copilot?sessionId=${item.session_id}`;
  }
  return `/team/${item.expert_id}`;
}

export function getStatusLabel(status: ExpertFirstJobStatus) {
  return STATUS_CONFIG[status].label;
}

export function getStatusConfig(status: ExpertFirstJobStatus) {
  return STATUS_CONFIG[status];
}

/** The server's detail names missing tools by slug; show their names. */
export function getFirstJobDetail(item: ExpertFirstJob) {
  const providers = item.providers ?? [];
  if (item.status === "needs_setup" && providers.length > 0) {
    return `Needs ${providers.map(serviceLabelFromIcon).join(", ")}`;
  }
  return item.detail;
}

/** Installing and working settle on their own; everything else waits on
 *  the user, so there is nothing to poll for. */
export function shouldPoll(items: ExpertFirstJob[]) {
  return items.some((item) => MOVING_STATUSES.includes(item.status));
}

export function trackFirstJobOpened(item: ExpertFirstJob) {
  trackFunnel("team_first_job_opened", {
    expert_id: item.expert_id,
    status: item.status,
  });
}
