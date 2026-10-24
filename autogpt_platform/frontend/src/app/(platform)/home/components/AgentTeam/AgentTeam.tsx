import { AiEraserIcon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { formatWeeklySpend } from "../../helpers";
import { HomeTileEmpty } from "../HomeTileEmpty/HomeTileEmpty";
import { HomeTile } from "../HomeTile/HomeTile";
import { AgentRow } from "./components/AgentRow";

// The tile lists this many experts; past that, its link offers the rest.
const LISTED_EXPERTS = 3;

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
        <div className="divide-y divide-separator">
          {agents.slice(0, LISTED_EXPERTS).map((agent) => (
            <AgentRow key={agent.expert.id} agent={agent} />
          ))}
        </div>
      )}

      {team.total > 0 ? (
        <Link
          href="/team"
          className="group flex items-center justify-between border-t border-separator px-4 py-2 outline-none transition-colors hover:bg-surface-muted focus-visible:bg-surface-muted"
        >
          <Text
            variant="small-medium"
            as="span"
            tone="muted"
            className="transition-colors group-hover:text-foreground-strong"
          >
            {team.total > LISTED_EXPERTS
              ? `View all ${team.total} experts`
              : "Open team"}
          </Text>
          <Icon icon={ArrowRight01Icon} size={14} aria-hidden="true" />
        </Link>
      ) : null}
    </HomeTile>
  );
}
