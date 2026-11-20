"use client";

import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
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
import type { IconSvgElement } from "@hugeicons/react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { RelativeTime } from "@/components/atoms/RelativeTime/RelativeTime";

const PENDING_COLORS =
  "border-warning-surface bg-warning-surface text-yellow-foreground";
const WARNING_COLORS =
  "border-warning-surface bg-warning-surface text-warning-foreground";

interface StatusLook {
  className: string;
  icon: IconSvgElement;
  /** Read out after "Task", e.g. "Task running". */
  text: string;
}

const STATUS_LOOKS: Record<AgentExecutionStatus, StatusLook> = {
  INCOMPLETE: {
    className: "border-error-surface bg-error-surface text-error-foreground",
    icon: AlertCircleIcon,
    text: "incomplete",
  },
  QUEUED: { className: PENDING_COLORS, icon: Clock01Icon, text: "queued" },
  RUNNING: {
    className: PENDING_COLORS,
    icon: PauseCircleIcon,
    text: "running",
  },
  REVIEW: {
    className: PENDING_COLORS,
    icon: PauseCircleIcon,
    text: "awaiting review",
  },
  COMPLETED: {
    className:
      "border-success-surface bg-success-surface text-success-foreground",
    icon: CheckmarkCircle02Icon,
    text: "completed",
  },
  TERMINATED: {
    className: "border-surface-muted bg-surface-muted text-foreground-body",
    icon: StopCircleIcon,
    text: "stopped",
  },
  FAILED: {
    className: "border-error-surface bg-error-surface text-error-foreground",
    icon: CancelCircleIcon,
    text: "failed",
  },
};

const COMPLETED_WITH_ERRORS: StatusLook = {
  className: WARNING_COLORS,
  icon: Alert02Icon,
  text: "completed with errors",
};

interface StatusIconProps {
  run: GraphExecutionMeta;
  withErrors: boolean;
}

function StatusIcon({ run, withErrors }: StatusIconProps) {
  const status = withErrors ? COMPLETED_WITH_ERRORS : STATUS_LOOKS[run.status];
  if (!status) return null;
  // A simulated run shows a flask whatever its status; its label says both.
  const look = run.is_dry_run
    ? { className: WARNING_COLORS, icon: FlaskConicalIcon }
    : status;
  return (
    <IconWrapper
      className={look.className}
      label={`${run.is_dry_run ? "Simulated task" : "Task"} ${status.text}`}
    >
      <Icon icon={look.icon} size={16} />
    </IconWrapper>
  );
}

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
  const withErrors = isCompletedWithErrors(run);
  const startedAgo = run.started_at ? (
    <RelativeTime date={run.started_at} />
  ) : (
    "—"
  );

  return (
    <SidebarItemCard
      icon={<StatusIcon run={run} withErrors={withErrors} />}
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
