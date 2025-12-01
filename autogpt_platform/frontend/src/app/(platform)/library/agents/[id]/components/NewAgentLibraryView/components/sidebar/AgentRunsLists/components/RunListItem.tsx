"use client";

import React from "react";
import moment from "moment";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { RunSidebarCard } from "./RunSidebarCard";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import {
  CheckCircleIcon,
  ClockIcon,
  PauseCircleIcon,
  StopCircleIcon,
  WarningCircleIcon,
  XCircleIcon,
} from "@phosphor-icons/react";
import { IconWrapper } from "./RunIconWrapper";

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
    <IconWrapper className="border-orange-50 bg-orange-50">
      <PauseCircleIcon size={16} className="text-orange-700" weight="bold" />
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

interface RunListItemProps {
  run: GraphExecutionMeta;
  title: string;
  selected?: boolean;
  onClick?: () => void;
}

export function RunListItem({
  run,
  title,
  selected,
  onClick,
}: RunListItemProps) {
  return (
    <RunSidebarCard
      icon={statusIconMap[run.status]}
      title={title}
      description={moment(run.started_at).fromNow()}
      onClick={onClick}
      selected={selected}
    />
  );
}
