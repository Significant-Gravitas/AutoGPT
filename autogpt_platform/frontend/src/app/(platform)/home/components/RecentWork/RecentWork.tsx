"use client";

import { WorkHistoryIcon } from "@hugeicons/core-free-icons";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { HomeTile } from "../HomeTile/HomeTile";
import { HomeTileEmpty } from "../HomeTileEmpty/HomeTileEmpty";
import { WorkGroup } from "./components/WorkGroup";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

export function RecentWork({ dashboard, className }: Props) {
  const groups = dashboard.recent_work?.groups ?? [];

  return (
    <HomeTile
      className={className}
      contentClassName="flex flex-col"
      surfaceClassName="py-4 sm:py-4"
      title={
        <div className="flex min-w-0 items-center gap-2">
          <Icon
            icon={WorkHistoryIcon}
            size={18}
            className="text-zinc-500"
            aria-hidden="true"
          />
          <Text variant="h5" className="text-zinc-950">
            Recent work
          </Text>
        </div>
      }
      header={
        <Text variant="large" className="text-zinc-600">
          What your agents produced this week.
        </Text>
      }
    >
      {groups.length === 0 ? (
        <HomeTileEmpty
          icon={WorkHistoryIcon}
          title="No work delivered yet"
          description="Files your agents write, integrations they use and schedules they set up will appear here."
        />
      ) : (
        <div className="-mx-4 divide-y divide-zinc-100 sm:-mx-5">
          {groups.map((group) => (
            <WorkGroup
              key={`${group.actor.name}-${group.session_id ?? "none"}`}
              group={group}
              timezone={dashboard.timezone}
            />
          ))}
        </div>
      )}
    </HomeTile>
  );
}
