"use client";

import {
  CheckmarkCircle02Icon,
  FilterHorizontalIcon,
  TaskDone01Icon,
} from "@hugeicons/core-free-icons";
import { useState } from "react";
import type { HomeBriefingOutcome } from "@/app/api/__generated__/models/homeBriefingOutcome";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { HomeTileExpandButton } from "../HomeTileExpandButton/HomeTileExpandButton";
import { HomeTile } from "../HomeTile/HomeTile";
import { OutcomeRow } from "./OutcomeRow";

interface Props {
  dashboard: HomeDashboardResponse;
  className?: string;
}

type BriefingFilter = "all" | HomeBriefingOutcome["status"];

const FILTER_LABELS: Record<BriefingFilter, string> = {
  all: "All",
  completed: "Completed",
  failed: "Failed",
};

export function MorningBriefing({ dashboard, className }: Props) {
  const { briefing } = dashboard;
  const [activeFilter, setActiveFilter] = useState<BriefingFilter>("all");
  const filterStatuses = Array.from(
    new Set(briefing.outcomes.map((outcome) => outcome.status)),
  );
  const selectedFilter: BriefingFilter =
    activeFilter !== "all" && filterStatuses.includes(activeFilter)
      ? activeFilter
      : "all";
  const visibleOutcomes =
    selectedFilter === "all"
      ? briefing.outcomes
      : briefing.outcomes.filter(
          (outcome) => outcome.status === selectedFilter,
        );

  return (
    <HomeTile
      className={className}
      contentClassName="flex flex-col gap-4"
      surfaceClassName="py-4 sm:py-4"
      title={
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <Icon
              icon={TaskDone01Icon}
              size={18}
              className="text-zinc-500"
              aria-hidden="true"
            />
            <Text variant="h5" className="text-zinc-950">
              Your briefing
            </Text>
          </div>
          <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
            <div className="flex items-center gap-3 text-xs font-medium tabular-nums text-zinc-500">
              <span>{briefing.completed_count} completed</span>
              {briefing.failed_count > 0 ? (
                <span className="text-rose-700">
                  {briefing.failed_count} failed
                </span>
              ) : null}
            </div>
            {filterStatuses.length > 1 ? (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="secondary"
                    size="small"
                    className="min-w-0"
                    leftIcon={
                      <Icon
                        icon={FilterHorizontalIcon}
                        size={15}
                        aria-hidden="true"
                      />
                    }
                    aria-label={`Filter briefing outcomes: ${FILTER_LABELS[selectedFilter]}`}
                    unmask={false}
                  >
                    {FILTER_LABELS[selectedFilter]}
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="min-w-36">
                  <DropdownMenuRadioGroup
                    value={selectedFilter}
                    onValueChange={(value) =>
                      setActiveFilter(value as BriefingFilter)
                    }
                  >
                    <DropdownMenuRadioItem value="all">
                      All
                    </DropdownMenuRadioItem>
                    {filterStatuses.map((status) => (
                      <DropdownMenuRadioItem key={status} value={status}>
                        {FILTER_LABELS[status]}
                      </DropdownMenuRadioItem>
                    ))}
                  </DropdownMenuRadioGroup>
                </DropdownMenuContent>
              </DropdownMenu>
            ) : null}
            <HomeTileExpandButton
              label="Open briefing activity"
              href="/library"
            />
          </div>
        </div>
      }
      header={
        <Text variant="large" className="text-zinc-600">
          The outcomes worth knowing from the last 24 hours.
        </Text>
      }
    >
      {briefing.outcomes.length === 0 ? (
        <div className="py-10 text-center">
          <Text variant="body-medium" className="text-zinc-800">
            No new outcomes yet
          </Text>
          <Text variant="small" className="mt-1 text-zinc-500">
            Completed work and useful exceptions will appear here.
          </Text>
        </div>
      ) : (
        <div className="-mx-4 divide-y divide-zinc-100 sm:-mx-5">
          {visibleOutcomes.map((outcome) => (
            <OutcomeRow key={outcome.id} outcome={outcome} />
          ))}
        </div>
      )}

      {briefing.routine_count > 0 ? (
        <div className="inline-flex items-center gap-1.5 self-end rounded-full border border-purple-500 bg-purple-100 px-2.5 py-1 text-sm font-medium text-purple-600">
          <Icon icon={CheckmarkCircle02Icon} size={15} aria-hidden="true" />
          Plus {briefing.routine_count} routine task
          {briefing.routine_count === 1 ? "" : "s"} completed quietly.
        </div>
      ) : null}
    </HomeTile>
  );
}
