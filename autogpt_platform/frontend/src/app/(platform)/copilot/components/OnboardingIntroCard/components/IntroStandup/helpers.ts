import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertSetupItem } from "@/app/api/__generated__/models/expertSetupItem";
import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";
import { safeHumanizeCronExpression } from "@/lib/cron-expression-utils";
import { getExpertRoleLabel } from "@/services/experts/expert-role-label";

export type MemberStatus =
  | "installing"
  | "unchecked"
  | "failed"
  | "needs_connection"
  | "needs_setup"
  | "working"
  | "ready";

export interface TeamMember {
  expert: Expert;
  area: string;
  /** A schedule the install created and that is not paused. */
  schedule: string | null;
  status: MemberStatus;
}

export type IntroAsk =
  | {
      kind: "connect";
      provider: string;
      service: string;
      names: string[];
      expertIds: string[];
    }
  | { kind: "dump" }
  | null;

export type ConnectAsk = Extract<NonNullable<IntroAsk>, { kind: "connect" }>;

export const VISIBLE_MEMBERS = 3;

export function buildTeam(args: {
  hires: Expert[];
  /** `null` when the setup gaps could not be read. */
  setupItems: ExpertSetupItem[] | null;
  /** Hires whose setup gaps are not known yet; `failed` once reading them has. */
  unchecked?: { ids: string[]; failed: boolean };
}): TeamMember[] {
  const gaps = getSetupGaps(args.setupItems ?? []);
  return args.hires.map((expert) => ({
    expert,
    area: expert.job_title || getExpertRoleLabel(expert.role),
    schedule: getRunningSchedule(expert),
    status: args.unchecked?.ids.includes(expert.id)
      ? args.unchecked.failed
        ? "unchecked"
        : "installing"
      : getMemberStatus(
          expert,
          args.setupItems ? (gaps.get(expert.id) ?? null) : "unknown",
        ),
  }));
}

type SetupGap = "connection" | "setup" | "unknown" | null;

export function getMemberStatus(expert: Expert, gap: SetupGap): MemberStatus {
  if (expert.setup_status === "installing") return "installing";
  if (expert.setup_status === "failed") return "failed";
  if (gap === "unknown") return "unchecked";
  if (gap === "connection") return "needs_connection";
  if (gap === "setup") return "needs_setup";
  const runStatus = expert.last_run_status?.toUpperCase();
  if (runStatus === "RUNNING" || runStatus === "QUEUED") return "working";
  return "ready";
}

export function getMemberStatusLabel(status: MemberStatus) {
  return {
    installing: "Setting up",
    unchecked: "Couldn't check setup",
    failed: "Setup didn't finish",
    needs_connection: "Needs a connection",
    needs_setup: "Needs setup",
    working: "Working",
    ready: "Ready",
  }[status];
}

// Every setup item is a scheduled workflow with no schedule yet, so each one
// leaves the hire work to do; a missing connection is named because Otto asks for it.
function getSetupGaps(setupItems: ExpertSetupItem[]) {
  const gaps = new Map<string, SetupGap>();
  setupItems.forEach((item) => {
    if (gaps.get(item.expert_id) === "connection") return;
    gaps.set(
      item.expert_id,
      item.resolution === "connect" ? "connection" : "setup",
    );
  });
  return gaps;
}

// A cron with no schedule id never got a schedule; the status names that gap.
function getRunningSchedule(expert: Expert) {
  if (expert.schedules_paused_at) return null;
  const workflow = expert.workflows.find(
    (workflow) => workflow.schedule_id && workflow.schedule_cron,
  );
  if (!workflow?.schedule_cron) return null;
  const cadence = safeHumanizeCronExpression(workflow.schedule_cron);
  const when = /^(Every|On day)\b/.test(cadence) ? cadence : "Scheduled";
  return workflow.name ? `${workflow.name} · ${when}` : when;
}

export function getTeamSummary(team: TeamMember[]) {
  const counts = new Map<MemberStatus, number>();
  team.forEach((member) =>
    counts.set(member.status, (counts.get(member.status) ?? 0) + 1),
  );
  const order: MemberStatus[] = [
    "working",
    "installing",
    "needs_connection",
    "needs_setup",
    "unchecked",
    "failed",
    "ready",
  ];
  return order
    .filter((status) => counts.has(status))
    .map(
      (status) =>
        `${counts.get(status)} ${getMemberStatusLabel(status).toLowerCase()}`,
    )
    .join(" · ");
}

export function chooseAsk(args: {
  path: string;
  team: TeamMember[];
  setupItems: ExpertSetupItem[];
  isConnectable: (provider: string) => boolean;
  connectDismissed: boolean;
}): IntroAsk {
  const connect = args.connectDismissed
    ? null
    : getConnectAsk(args.setupItems, args.team, args.isConnectable);
  if (connect) return connect;
  // Path A's next steps are the suggested tasks, written from the dump.
  if (args.path === "B") return { kind: "dump" };
  return null;
}

// The service the most hires are waiting on, so one connection unblocks
// as much of the team as possible.
function getConnectAsk(
  setupItems: ExpertSetupItem[],
  team: TeamMember[],
  isConnectable: (provider: string) => boolean,
): IntroAsk {
  const onTeam = new Map(team.map((member) => [member.expert.id, member]));
  const byProvider = new Map<string, Set<string>>();
  setupItems.forEach((item) => {
    const provider = item.providers.find(isConnectable);
    if (item.resolution !== "connect" || !provider) return;
    if (!onTeam.has(item.expert_id)) return;
    const ids = byProvider.get(provider) ?? new Set<string>();
    ids.add(item.expert_id);
    byProvider.set(provider, ids);
  });
  const [best] = Array.from(byProvider.entries()).sort(
    (a, b) => b[1].size - a[1].size,
  );
  if (!best) return null;
  const [provider, ids] = best;
  const expertIds = Array.from(ids);
  return {
    kind: "connect",
    provider,
    service: formatProviderName(provider),
    expertIds,
    names: expertIds.map((id) => onTeam.get(id)?.expert.name ?? ""),
  };
}

export function joinNames(names: string[]) {
  if (names.length <= 1) return names[0] ?? "";
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
}
