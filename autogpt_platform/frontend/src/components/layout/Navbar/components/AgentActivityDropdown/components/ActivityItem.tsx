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
import { useRouter } from "next/navigation";
import type { AgentExecutionWithInfo } from "../helpers";
import { formatTimeAgo, getExecutionDuration } from "../helpers";

interface Props {
  execution: AgentExecutionWithInfo;
}

export function ActivityItem({ execution }: Props) {
  const router = useRouter();

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

  return (
    <div
      className="cursor-pointer border-b border-slate-50 px-2 py-3 transition-colors last:border-b-0 hover:bg-lightGrey"
      onClick={() => {
        const agentId = execution.library_agent_id || execution.graph_id;
        router.push(`/library/agents/${agentId}?executionId=${execution.id}`);
      }}
      role="button"
    >
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
    </div>
  );
}
