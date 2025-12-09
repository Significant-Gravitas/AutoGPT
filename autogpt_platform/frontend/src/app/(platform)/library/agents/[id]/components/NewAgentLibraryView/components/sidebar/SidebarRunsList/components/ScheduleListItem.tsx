"use client";

import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { ClockClockwiseIcon } from "@phosphor-icons/react";
import moment from "moment";
import { IconWrapper } from "./RunIconWrapper";
import { RunSidebarCard } from "./RunSidebarCard";

interface ScheduleListItemProps {
  schedule: GraphExecutionJobInfo;
  selected?: boolean;
  onClick?: () => void;
}

export function ScheduleListItem({
  schedule,
  selected,
  onClick,
}: ScheduleListItemProps) {
  return (
    <RunSidebarCard
      title={schedule.name}
      description={moment(schedule.next_run_time).fromNow()}
      onClick={onClick}
      selected={selected}
      icon={
        <IconWrapper className="border-slate-50 bg-yellow-50">
          <ClockClockwiseIcon
            size={16}
            className="text-yellow-700"
            weight="bold"
          />
        </IconWrapper>
      }
    />
  );
}
