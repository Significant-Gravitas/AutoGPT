"use client";

import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import {
  CheckCircleIcon,
  ClockIcon,
  PauseCircleIcon,
  StopCircleIcon,
  WarningCircleIcon,
  XCircleIcon,
} from "@phosphor-icons/react";
import moment from "moment";
import React from "react";
import { IconWrapper } from "./IconWrapper";
import { SidebarItemCard } from "./SidebarItemCard";
import { TaskActionsDropdown } from "./TaskActionsDropdown";

const statusIconMap: Record<AgentExecutionStatus, React.ReactNode> = {
  INCOMPLETE: (
    <IconWrapper className="border-red-50 bg-red-50">
      <WarningCircleIcon size={16} className="text-red-700" weight="bold" />
    </IconWrapper>
  ),
  QUEUED: (
    <IconWrapper className="border-yellow-50 bg-yellow-50">
      <ClockIcon size={16} className="text-yellow-700" weight="bold" />
    </IconWrapper>
  ),
  RUNNING: (
    <IconWrapper className="border-yellow-50 bg-yellow-50">
      <PauseCircleIcon size={16} className="text-yellow-700" weight="bold" />
    </IconWrapper>
  ),
  REVIEW: (
    <IconWrapper className="border-yellow-50 bg-yellow-50">
      <PauseCircleIcon size={16} className="text-yellow-700" weight="bold" />
    </IconWrapper>
  ),
  COMPLETED: (
    <IconWrapper className="border-green-50 bg-green-50">
      <CheckCircleIcon size={16} className="text-green-700" weight="bold" />
    </IconWrapper>
  ),
  TERMINATED: (
    <IconWrapper className="border-slate-50 bg-slate-50">
      <StopCircleIcon size={16} className="text-slate-700" weight="bold" />
    </IconWrapper>
  ),
  FAILED: (
    <IconWrapper className="border-red-50 bg-red-50">
      <XCircleIcon size={16} className="text-red-700" weight="bold" />
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
  return (
    <SidebarItemCard
      icon={statusIconMap[run.status]}
      title={title}
      description={moment(run.started_at).fromNow()}
      onClick={onClick}
      selected={selected}
      actions={
        <TaskActionsDropdown agent={agent} run={run} onDeleted={onDeleted} />
      }
    />
  );
}
