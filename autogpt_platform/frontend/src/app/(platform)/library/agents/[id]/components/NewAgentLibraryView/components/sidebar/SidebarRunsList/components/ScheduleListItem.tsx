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
  onRunCreated?: (runID: string) => void;
}

export function ScheduleListItem({
  schedule,
  agent,
  selected,
  onClick,
  onDeleted,
  onRunCreated,
}: Props) {
  const isPaused = Boolean(schedule.is_paused);
  const pausedLabel =
    schedule.paused_reason === "payment_lapsed"
      ? "Paused — payment required"
      : "Paused";
  const description = isPaused
    ? pausedLabel
    : schedule.next_run_time
      ? formatDistanceToNow(schedule.next_run_time, {
          addSuffix: true,
        })
      : "Pending";
  return (
    <SidebarItemCard
      title={schedule.name}
      description={description}
      descriptionTitle={
        !isPaused && schedule.next_run_time
          ? new Date(schedule.next_run_time).toString()
          : description
      }
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
          onRunCreated={onRunCreated}
        />
      }
    />
  );
}
