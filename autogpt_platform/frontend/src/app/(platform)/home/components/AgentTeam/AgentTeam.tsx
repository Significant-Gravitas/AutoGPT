import { AiEraserIcon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { formatWeeklySpend } from "../../helpers";
import { HomeTileEmpty } from "../HomeTileEmpty/HomeTileEmpty";
import { HomeTile } from "../HomeTile/HomeTile";
import { AgentRow } from "./components/AgentRow";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

export function AgentTeam({ dashboard, className }: Props) {
  const { team, agents } = dashboard;
  const teamSpend = formatWeeklySpend(team.spend_cents);
  const statusLine =
    team.working > 0
      ? `${team.working} working now · ${team.ready} ready`
      : `${team.ready} ready for work`;

  return (
    <HomeTile
      className={className}
      icon={AiEraserIcon}
      title="Your team"
      meta={
        team.total > 0 ? (
          <span className="tabular-nums">
            {teamSpend ? `${statusLine} · ${teamSpend}` : statusLine}
          </span>
        ) : null
      }
    >
      {agents.length === 0 ? (
        <HomeTileEmpty
          title="Build your team"
          description="Hire an expert to start delegating work."
          action={{ href: "/marketplace", label: "Browse experts" }}
        />
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
          className="flex items-center justify-between border-t border-zinc-100 px-4 py-2 text-xs font-medium text-zinc-500 outline-none transition-colors hover:bg-zinc-50 hover:text-zinc-900 focus-visible:bg-zinc-50"
        >
          View all {team.total} experts
          <Icon icon={ArrowRight01Icon} size={14} aria-hidden="true" />
        </Link>
      ) : null}
    </HomeTile>
  );
}
