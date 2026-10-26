"use client";

import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import React from "react";
import {
  describeFailedSteps,
  getFailedStepCount,
  isCompletedWithErrors,
} from "@/lib/run-health";
import { IconWrapper } from "./IconWrapper";
import { SidebarItemCard } from "./SidebarItemCard";
import { TaskActionsDropdown } from "./TaskActionsDropdown";
import {
  Alert02Icon,
  AlertCircleIcon,
  CancelCircleIcon,
  CheckmarkCircle02Icon,
  Clock01Icon,
  FlaskConicalIcon,
  PauseCircleIcon,
  StopCircleIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { RelativeTime } from "@/components/atoms/RelativeTime/RelativeTime";

const statusIconMap: Record<AgentExecutionStatus, React.ReactNode> = {
  INCOMPLETE: (
    <IconWrapper className="border-error-surface bg-error-surface">
      <Icon
        icon={AlertCircleIcon}
        size={16}
        className="text-error-foreground"
      />
    </IconWrapper>
  ),
  QUEUED: (
    <IconWrapper
      // TODO(tokens): no exact role for border-yellow-50 (#FEF9E6), bg-yellow-50 (#FEF9E6)
      className="border-yellow-50 bg-yellow-50"
    >
      <Icon
        // TODO(tokens): no exact role for text-yellow-700 (#AE8800)
        icon={Clock01Icon}
        size={16}
        className="text-yellow-700"
      />
    </IconWrapper>
  ),
  RUNNING: (
    <IconWrapper
      // TODO(tokens): no exact role for border-yellow-50 (#FEF9E6), bg-yellow-50 (#FEF9E6)
      className="border-yellow-50 bg-yellow-50"
    >
      <Icon
        // TODO(tokens): no exact role for text-yellow-700 (#AE8800)
        icon={PauseCircleIcon}
        size={16}
        className="text-yellow-700"
      />
    </IconWrapper>
  ),
  REVIEW: (
    <IconWrapper
      // TODO(tokens): no exact role for border-yellow-50 (#FEF9E6), bg-yellow-50 (#FEF9E6)
      className="border-yellow-50 bg-yellow-50"
    >
      <Icon
        // TODO(tokens): no exact role for text-yellow-700 (#AE8800)
        icon={PauseCircleIcon}
        size={16}
        className="text-yellow-700"
      />
    </IconWrapper>
  ),
  COMPLETED: (
    <IconWrapper className="border-success-surface bg-success-surface">
      <Icon
        icon={CheckmarkCircle02Icon}
        size={16}
        className="text-success-foreground"
      />
    </IconWrapper>
  ),
  TERMINATED: (
    <IconWrapper
      // TODO(tokens): no exact role for border-slate-50 (#F7F8F9), bg-slate-50 (#F7F8F9)
      className="border-slate-50 bg-slate-50"
    >
      <Icon
        // TODO(tokens): no exact role for text-slate-700 (#475263)
        icon={StopCircleIcon}
        size={16}
        className="text-slate-700"
      />
    </IconWrapper>
  ),
  FAILED: (
    <IconWrapper className="border-error-surface bg-error-surface">
      <Icon
        icon={CancelCircleIcon}
        size={16}
        className="text-error-foreground"
      />
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

const completedWithErrorsIcon = (
  <IconWrapper className="border-warning-surface bg-warning-surface">
    <Icon icon={Alert02Icon} size={16} className="text-warning-foreground" />
  </IconWrapper>
);

export function TaskListItem({
  run,
  title,
  agent,
  selected,
  onClick,
  onDeleted,
}: Props) {
  const withErrors = isCompletedWithErrors(run);
  const icon = run.is_dry_run ? (
    <IconWrapper className="border-warning-surface bg-warning-surface">
      <Icon
        icon={FlaskConicalIcon}
        size={16}
        className="text-warning-foreground"
      />
    </IconWrapper>
  ) : withErrors ? (
    completedWithErrorsIcon
  ) : (
    statusIconMap[run.status]
  );
  const startedAgo = run.started_at ? (
    <RelativeTime date={run.started_at} />
  ) : (
    "—"
  );

  return (
    <SidebarItemCard
      icon={icon}
      title={run.is_dry_run ? `${title} (Simulated)` : title}
      description={
        withErrors ? (
          <>
            {describeFailedSteps(getFailedStepCount(run))} · {startedAgo}
          </>
        ) : (
          startedAgo
        )
      }
      descriptionTitle={
        run.started_at ? new Date(run.started_at).toString() : undefined
      }
      onClick={onClick}
      selected={selected}
      testID={`library-run-item-${run.id}`}
      actions={
        <TaskActionsDropdown agent={agent} run={run} onDeleted={onDeleted} />
      }
    />
  );
}
