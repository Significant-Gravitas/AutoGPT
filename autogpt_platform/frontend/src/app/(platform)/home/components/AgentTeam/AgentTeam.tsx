import {
  AlertCircleIcon,
  ArrowRight01Icon,
  CheckmarkCircle02Icon,
  HelpCircleIcon,
  Loading03Icon,
  PauseIcon,
  Settings01Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { HomeTile } from "../HomeTile/HomeTile";
import { formatUntil } from "../NowNext/helpers";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

export function AgentTeam({ dashboard, className }: Props) {
  const { team, agents } = dashboard;
  const readyCount = team.ready + team.working;
  return (
    <HomeTile
      className={className}
      contentClassName="flex flex-col"
      title={
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <Icon
              icon={UserGroupIcon}
              size={18}
              className="text-zinc-500"
              aria-hidden="true"
            />
            <Text variant="h5" className="text-zinc-950">
              Your agents
            </Text>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            {team.total > 0 ? (
              <span className="shrink-0 rounded-md bg-zinc-100 px-2 py-1 text-xs font-medium tabular-nums text-zinc-600">
                {readyCount}/{team.total} available
              </span>
            ) : null}
          </div>
        </div>
      }
      header={
        <Text variant="large" className="text-zinc-600">
          {team.working > 0
            ? `${team.working} working now · ${team.ready} ready`
            : `${team.ready} ready for work`}
        </Text>
      }
    >
      {agents.length === 0 ? (
        <EmptyTeam />
      ) : (
        <div className="divide-y divide-zinc-100">
          {agents.slice(0, 3).map((agent) => (
            <AgentRow key={agent.expert.id} agent={agent} />
          ))}
        </div>
      )}

      {team.total > 0 ? (
        <Link
          href="/team"
          className="mt-2 flex items-center justify-between px-1 text-sm font-medium text-zinc-600 transition-colors hover:text-zinc-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
        >
          View all {team.total} agents
          <Icon icon={ArrowRight01Icon} size={16} aria-hidden="true" />
        </Link>
      ) : null}
    </HomeTile>
  );
}

function AgentRow({ agent }: { agent: HomeAgentStatus }) {
  return (
    <Link
      href={`/team/${agent.expert.id}`}
      className="group -mx-2 flex min-w-0 items-center gap-3 rounded-lg px-2 py-2.5 transition-colors hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
    >
      <ExpertAvatar
        name={agent.expert.name}
        avatarUrl={agent.expert.avatar_url}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <Text variant="body-medium" className="truncate text-zinc-900">
            {agent.expert.name}
          </Text>
          <StatusBadge status={agent.status} />
        </div>
        <Text variant="small" className="truncate text-zinc-500">
          {agent.detail}
        </Text>
      </div>
      {agent.next_run_time ? (
        <Text
          variant="small"
          className="shrink-0 tabular-nums text-zinc-400 group-hover:text-zinc-600"
        >
          {formatUntil(agent.next_run_time)}
        </Text>
      ) : (
        <Icon
          icon={ArrowRight01Icon}
          size={15}
          className="shrink-0 text-zinc-300 transition-transform group-hover:translate-x-0.5 group-hover:text-zinc-600"
          aria-hidden="true"
        />
      )}
    </Link>
  );
}

const STATUS_CONFIG = {
  working: {
    label: "Working",
    icon: Loading03Icon,
    className: "bg-primary/10 text-primary",
  },
  ready: {
    label: "Ready",
    icon: CheckmarkCircle02Icon,
    className: "bg-emerald-50 text-emerald-700",
  },
  needs_setup: {
    label: "Setup",
    icon: Settings01Icon,
    className: "bg-amber-50 text-amber-700",
  },
  paused: {
    label: "Paused",
    icon: PauseIcon,
    className: "bg-zinc-100 text-zinc-600",
  },
  failed: {
    label: "Failed",
    icon: AlertCircleIcon,
    className: "bg-rose-50 text-rose-700",
  },
};

const UNKNOWN_STATUS_CONFIG = {
  label: "Unknown",
  icon: HelpCircleIcon,
  className: "bg-zinc-100 text-zinc-600",
};

function StatusBadge({ status }: { status: HomeAgentStatus["status"] }) {
  const config = STATUS_CONFIG[status] ?? UNKNOWN_STATUS_CONFIG;

  return (
    <span
      className={cn(
        "inline-flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-xs font-medium",
        config.className,
      )}
    >
      <Icon
        icon={config.icon}
        size={13}
        className={cn(status === "working" && "animate-spin")}
        aria-hidden="true"
      />
      {config.label}
    </span>
  );
}

function EmptyTeam() {
  return (
    <div className="rounded-lg border border-dashed border-zinc-200 p-5 text-center">
      <Text variant="body-medium">Build your team</Text>
      <Text variant="small" className="mt-1 text-zinc-500">
        Hire an expert to start delegating work.
      </Text>
      <Link
        href="/marketplace"
        className="mt-3 inline-block text-sm font-medium text-zinc-700 hover:text-zinc-950 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400"
      >
        Browse experts
      </Link>
    </div>
  );
}
