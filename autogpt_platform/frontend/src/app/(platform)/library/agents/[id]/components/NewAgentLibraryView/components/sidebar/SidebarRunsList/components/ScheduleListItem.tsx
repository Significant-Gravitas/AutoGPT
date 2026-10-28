"use client";

import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { formatDistanceToNow } from "date-fns";
import { IconWrapper } from "./IconWrapper";
import { ScheduleActionsDropdown } from "./ScheduleActionsDropdown";
import { SidebarItemCard } from "./SidebarItemCard";
import { TimeScheduleIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
  return (
    <SidebarItemCard
      title={schedule.name}
      description={formatDistanceToNow(schedule.next_run_time, {
        addSuffix: true,
      })}
      descriptionTitle={new Date(schedule.next_run_time).toString()}
      onClick={onClick}
      selected={selected}
      icon={
        <IconWrapper
          // TODO(tokens): no exact role for border-slate-50 (#F7F8F9), bg-yellow-50 (#FEF9E6)
          className="border-slate-50 bg-yellow-50"
        >
          <Icon
            // TODO(tokens): no exact role for text-yellow-700 (#AE8800)
            icon={TimeScheduleIcon}
            size={16}
            className="text-yellow-700"
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
