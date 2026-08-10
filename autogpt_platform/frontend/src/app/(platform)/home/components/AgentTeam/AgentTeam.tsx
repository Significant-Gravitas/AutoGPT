import { ArrowRight01Icon, UserGroupIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { HomeTile } from "../HomeTile/HomeTile";
import { AgentRow } from "./components/AgentRow";
import { EmptyTeam } from "./components/EmptyTeam";

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
