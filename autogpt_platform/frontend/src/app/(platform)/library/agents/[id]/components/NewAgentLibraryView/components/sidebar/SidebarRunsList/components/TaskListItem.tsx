"use client";

import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { formatDistanceToNow } from "date-fns";
import React from "react";
import { IconWrapper } from "./IconWrapper";
import { SidebarItemCard } from "./SidebarItemCard";
import { TaskActionsDropdown } from "./TaskActionsDropdown";
import {
  AlertCircleIcon,
  CancelCircleIcon,
  CheckmarkCircle02Icon,
  Clock01Icon,
  FlaskConicalIcon,
  PauseCircleIcon,
  StopCircleIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

const statusIconMap: Record<AgentExecutionStatus, React.ReactNode> = {
  INCOMPLETE: (
    <IconWrapper className="border-red-50 bg-red-50">
      <Icon icon={AlertCircleIcon} size={16} className="text-red-700" />
    </IconWrapper>
  ),
  QUEUED: (
    <IconWrapper className="border-yellow-50 bg-yellow-50">
      <Icon icon={Clock01Icon} size={16} className="text-yellow-700" />
    </IconWrapper>
  ),
  RUNNING: (
    <IconWrapper className="border-yellow-50 bg-yellow-50">
      <Icon icon={PauseCircleIcon} size={16} className="text-yellow-700" />
    </IconWrapper>
  ),
  REVIEW: (
    <IconWrapper className="border-yellow-50 bg-yellow-50">
      <Icon icon={PauseCircleIcon} size={16} className="text-yellow-700" />
    </IconWrapper>
  ),
  COMPLETED: (
    <IconWrapper className="border-green-50 bg-green-50">
      <Icon icon={CheckmarkCircle02Icon} size={16} className="text-green-700" />
    </IconWrapper>
  ),
  TERMINATED: (
    <IconWrapper className="border-slate-50 bg-slate-50">
      <Icon icon={StopCircleIcon} size={16} className="text-slate-700" />
    </IconWrapper>
  ),
  FAILED: (
    <IconWrapper className="border-red-50 bg-red-50">
      <Icon icon={CancelCircleIcon} size={16} className="text-red-700" />
    </IconWrapper>
  ),
};

interface Props {
  run: GraphExecutionMeta;
  title: string;
  agent: LibraryAgent;
  selected?: boolean;
  onClick?: () => void;
  onDeleted?: () => void;
}

export function TaskListItem({
  run,
  title,
  agent,
  selected,
  onClick,
  onDeleted,
}: Props) {
  const icon = run.is_dry_run ? (
    <IconWrapper className="border-amber-50 bg-amber-50">
      <Icon icon={FlaskConicalIcon} size={16} className="text-amber-700" />
    </IconWrapper>
  ) : (
    statusIconMap[run.status]
  );

  return (
    <SidebarItemCard
      icon={icon}
      title={run.is_dry_run ? `${title} (Simulated)` : title}
      description={
        run.started_at
          ? formatDistanceToNow(run.started_at, { addSuffix: true })
          : "—"
      }
      descriptionTitle={
        run.started_at ? new Date(run.started_at).toString() : undefined
      }
      onClick={onClick}
      selected={selected}
      actions={
        <TaskActionsDropdown agent={agent} run={run} onDeleted={onDeleted} />
      }
    />
  );
}
