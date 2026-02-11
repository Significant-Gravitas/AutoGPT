"use client";

import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { ClockClockwiseIcon } from "@phosphor-icons/react";
import { formatDistanceToNow } from "date-fns";
import { IconWrapper } from "./IconWrapper";
import { ScheduleActionsDropdown } from "./ScheduleActionsDropdown";
import { SidebarItemCard } from "./SidebarItemCard";

interface Props {
  schedule: GraphExecutionJobInfo;
  agent: LibraryAgent;
  selected?: boolean;
  onClick?: () => void;
  onDeleted?: () => void;
}

export function ScheduleListItem({
  schedule,
  agent,
  selected,
  onClick,
  onDeleted,
}: Props) {
  return (
    <SidebarItemCard
      title={schedule.name}
      description={formatDistanceToNow(schedule.next_run_time, {
        addSuffix: true,
      })}
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
      actions={
        <ScheduleActionsDropdown
          agent={agent}
          schedule={schedule}
          onDeleted={onDeleted}
        />
      }
    />
  );
}
