"use client";

import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { Text } from "@/components/atoms/Text/Text";
import {
  CheckCircle,
  CircleNotchIcon,
  Clock,
  WarningOctagonIcon,
  StopCircle,
  CircleDashed,
} from "@phosphor-icons/react";
import type { AgentExecutionWithInfo } from "../helpers";
import { getExecutionDuration } from "../helpers";
import Link from "next/link";
import { formatTimeAgo } from "@/lib/utils/time";

interface Props {
  execution: AgentExecutionWithInfo;
}

export function ActivityItem({ execution }: Props) {
  function getStatusIcon() {
    switch (execution.status) {
      case AgentExecutionStatus.QUEUED:
        return <Clock size={18} className="text-purple-500" />;
      case AgentExecutionStatus.RUNNING:
        return (
          <CircleNotchIcon
            size={18}
            className="animate-spin text-purple-500"
            weight="bold"
          />
        );
      case AgentExecutionStatus.COMPLETED:
        return (
          <CheckCircle size={18} weight="fill" className="text-purple-500" />
        );
      case AgentExecutionStatus.FAILED:
        return <WarningOctagonIcon size={18} className="text-purple-500" />;
      case AgentExecutionStatus.TERMINATED:
        return (
          <StopCircle size={18} className="text-purple-500" weight="fill" />
        );
      case AgentExecutionStatus.INCOMPLETE:
        return <CircleDashed size={18} className="text-purple-500" />;
      default:
        return null;
    }
  }

  function getTimeDisplay() {
    const isActiveStatus =
      execution.status === AgentExecutionStatus.RUNNING ||
      execution.status === AgentExecutionStatus.QUEUED;

    if (isActiveStatus) {
      const timeAgo = formatTimeAgo(execution.started_at.toString());
      const statusText =
        execution.status === AgentExecutionStatus.QUEUED ? "queued" : "running";
      return `Started ${timeAgo}, ${getExecutionDuration(execution)} ${statusText}`;
    }

    if (execution.ended_at) {
      const timeAgo = formatTimeAgo(execution.ended_at.toString());
      switch (execution.status) {
        case AgentExecutionStatus.COMPLETED:
          return `Completed ${timeAgo}`;
        case AgentExecutionStatus.FAILED:
          return `Failed ${timeAgo}`;
        case AgentExecutionStatus.TERMINATED:
          return `Stopped ${timeAgo}`;
        case AgentExecutionStatus.INCOMPLETE:
          return `Incomplete ${timeAgo}`;
        default:
          return `Ended ${timeAgo}`;
      }
    }

    return "Unknown";
  }

  const linkUrl = `/library/agents/${execution.library_agent_id}?executionId=${execution.id}`;
  const withExecutionLink = execution.library_agent_id && execution.id;

  const content = (
    <>
      {/* Icon + Agent Name */}
      <div className="flex items-center space-x-2">
        {getStatusIcon()}
        <Text
          variant="body-medium"
          className="max-w-[16rem] truncate text-gray-900"
        >
          {execution.agent_name}
        </Text>
      </div>

      {/* Agent Message - Indented */}
      <div className="ml-7 pt-1">
        {/* Time - Indented */}
        <Text variant="small" className="!text-zinc-500">
          {getTimeDisplay()}
        </Text>
      </div>
    </>
  );

  return withExecutionLink ? (
    <Link
      className="block cursor-pointer border-b border-slate-50 px-2 py-3 transition-colors last:border-b-0 hover:bg-lightGrey"
      href={linkUrl}
      role="button"
    >
      {content}
    </Link>
  ) : (
    <div className="block border-b border-slate-50 px-2 py-3 last:border-b-0">
      {content}
    </div>
  );
}
