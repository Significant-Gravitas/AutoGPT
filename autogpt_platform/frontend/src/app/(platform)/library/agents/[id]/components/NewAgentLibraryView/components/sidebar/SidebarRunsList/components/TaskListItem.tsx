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

const PENDING_COLORS =
  // oxlint-disable-next-line design-system/no-raw-colors -- yellow-700 #AE8800 has no role; mustard to orange-brown change deferred (see PR)
  "border-warning-surface bg-warning-surface text-yellow-700";

const statusIconMap: Record<AgentExecutionStatus, React.ReactNode> = {
  INCOMPLETE: (
    <IconWrapper className="border-error-surface bg-error-surface text-error-foreground">
      <Icon icon={AlertCircleIcon} size={16} />
    </IconWrapper>
  ),
  QUEUED: (
    <IconWrapper className={PENDING_COLORS}>
      <Icon icon={Clock01Icon} size={16} />
    </IconWrapper>
  ),
  RUNNING: (
    <IconWrapper className={PENDING_COLORS}>
      <Icon icon={PauseCircleIcon} size={16} />
    </IconWrapper>
  ),
  REVIEW: (
    <IconWrapper className={PENDING_COLORS}>
      <Icon icon={PauseCircleIcon} size={16} />
    </IconWrapper>
  ),
  COMPLETED: (
    <IconWrapper className="border-success-surface bg-success-surface text-success-foreground">
      <Icon icon={CheckmarkCircle02Icon} size={16} />
    </IconWrapper>
  ),
  TERMINATED: (
    <IconWrapper className="border-surface-muted bg-surface-muted text-foreground-body">
      <Icon icon={StopCircleIcon} size={16} />
    </IconWrapper>
  ),
  FAILED: (
    <IconWrapper className="border-error-surface bg-error-surface text-error-foreground">
      <Icon icon={CancelCircleIcon} size={16} />
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
  <IconWrapper className="border-warning-surface bg-warning-surface text-warning-foreground">
    <Icon icon={Alert02Icon} size={16} />
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
    <IconWrapper className="border-warning-surface bg-warning-surface text-warning-foreground">
      <Icon icon={FlaskConicalIcon} size={16} />
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
