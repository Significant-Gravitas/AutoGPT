"use client";

import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { Text } from "@/components/atoms/Text/Text";
import {
  CheckCircle,
  CircleNotchIcon,
  Clock,
  WarningOctagonIcon,
} from "@phosphor-icons/react";
import { useRouter } from "next/navigation";
import type { AgentExecutionWithInfo } from "../helpers";
import {
  formatTimeAgo,
  getExecutionDuration,
  getStatusColorClass,
} from "../helpers";

interface NotificationItemProps {
  execution: AgentExecutionWithInfo;
  type: "running" | "completed" | "failed";
}

export function NotificationItem({ execution, type }: NotificationItemProps) {
  const router = useRouter();

  function getStatusIcon() {
    switch (type) {
      case "running":
        return execution.status === AgentExecutionStatus.QUEUED ? (
          <Clock size={16} className="text-purple-500" />
        ) : (
          <CircleNotchIcon
            size={16}
            className="animate-spin text-purple-500"
            weight="bold"
          />
        );
      case "completed":
        return (
          <CheckCircle size={16} weight="fill" className="text-purple-500" />
        );
      case "failed":
        return <WarningOctagonIcon size={16} className="text-purple-500" />;
      default:
        return null;
    }
  }

  function getTimeDisplay() {
    if (type === "running") {
      const timeAgo = formatTimeAgo(execution.started_at.toString());
      return `Started ${timeAgo}, ${getExecutionDuration(execution)} running`;
    }

    if (execution.ended_at) {
      const timeAgo = formatTimeAgo(execution.ended_at.toString());
      return type === "completed"
        ? `Completed ${timeAgo}`
        : `Failed ${timeAgo}`;
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
      <div className="flex items-center space-x-3">
        {getStatusIcon()}
        <Text variant="body-medium" className="truncate text-gray-900">
          {execution.agent_name}
        </Text>
      </div>

      {/* Agent Message - Indented */}
      <div className="ml-7">
        {execution.agent_description ? (
          <Text variant="body" className={`${getStatusColorClass(execution)}`}>
            {execution.agent_description}
          </Text>
        ) : null}

        {/* Time - Indented */}
        <Text variant="small" className="pt-2 !text-zinc-500">
          {getTimeDisplay()}
        </Text>
      </div>
    </div>
  );
}
