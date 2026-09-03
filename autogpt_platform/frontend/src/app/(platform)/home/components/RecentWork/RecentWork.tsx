"use client";

import { WorkHistoryIcon } from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { HomeTile } from "../HomeTile/HomeTile";
import { HomeTileEmpty } from "../HomeTileEmpty/HomeTileEmpty";
import { WorkGroup } from "./components/WorkGroup";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

export function RecentWork({ dashboard, className }: Props) {
  const groups = dashboard.recent_work?.groups ?? [];
  const totalCount = dashboard.recent_work?.total_count ?? 0;

  return (
    <HomeTile
      className={className}
      icon={WorkHistoryIcon}
      title="Recent work"
      meta={
        totalCount > 0 ? (
          <span className="tabular-nums">{totalCount} this week</span>
        ) : null
      }
    >
      {groups.length === 0 ? (
        <HomeTileEmpty
          icon={WorkHistoryIcon}
          title="No work delivered yet"
          description="Files your agents write, integrations they use and schedules they set up will appear here."
        />
      ) : (
        <div className="divide-y divide-zinc-100">
          {groups.map((group) => (
            <WorkGroup
              key={group.items[0]?.id ?? group.actor.name}
              group={group}
              timezone={dashboard.timezone}
            />
          ))}
        </div>
      )}
    </HomeTile>
  );
}
