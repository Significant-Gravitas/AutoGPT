"use client";

import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { Text } from "@/components/atoms/Text/Text";
import { formatTimeAgo } from "@/lib/utils/time";
import {
  CheckCircleIcon,
  ClockIcon,
  StopCircleIcon,
  WarningIcon,
  SpinnerIcon,
  MinusCircleIcon,
} from "@phosphor-icons/react";
import Link from "next/link";
import type { AgentExecutionWithInfo } from "../helpers";
import { getExecutionDuration } from "../helpers";

interface Props {
  execution: AgentExecutionWithInfo;
}

export function ActivityItem({ execution }: Props) {
  function getStatusIcon() {
    switch (execution.status) {
      case AgentExecutionStatus.QUEUED:
        return <ClockIcon size={18} className="text-purple-500" />;
      case AgentExecutionStatus.RUNNING:
        return (
          <SpinnerIcon size={18} className="animate-spin text-purple-500" />
        );
      case AgentExecutionStatus.COMPLETED:
        return <CheckCircleIcon size={18} className="text-purple-500" />;
      case AgentExecutionStatus.FAILED:
        return <WarningIcon size={18} className="text-purple-500" />;
      case AgentExecutionStatus.TERMINATED:
        return <StopCircleIcon size={18} className="text-purple-500" />;
      case AgentExecutionStatus.INCOMPLETE:
        return <MinusCircleIcon size={18} className="text-purple-500" />;
      case AgentExecutionStatus.REVIEW:
        return <WarningIcon size={18} className="text-yellow-600" />;
      default:
        return null;
    }
  }

  function getItemDisplay() {
    // Handle active statuses (running/queued)
    const isActiveStatus =
      execution.status === AgentExecutionStatus.RUNNING ||
      execution.status === AgentExecutionStatus.QUEUED;

    if (isActiveStatus) {
      const timeAgo = formatTimeAgo(execution.started_at.toString());
      const statusText =
        execution.status === AgentExecutionStatus.QUEUED ? "queued" : "running";
      return [
        `Started ${timeAgo}, ${getExecutionDuration(execution)} ${statusText}`,
      ];
    }

    // Handle all other statuses with time display
    const timeAgo = execution.ended_at
      ? formatTimeAgo(execution.ended_at.toString())
      : formatTimeAgo(execution.started_at.toString());

    let statusText = "ended";
    switch (execution.status) {
      case AgentExecutionStatus.COMPLETED:
        statusText = "completed";
        break;
      case AgentExecutionStatus.FAILED:
        statusText = "failed";
        break;
      case AgentExecutionStatus.TERMINATED:
        statusText = "stopped";
        break;
      case AgentExecutionStatus.INCOMPLETE:
        statusText = "incomplete";
        break;
      case AgentExecutionStatus.REVIEW:
        statusText = "awaiting approval";
        break;
    }

    return [
      `${statusText.charAt(0).toUpperCase() + statusText.slice(1)} ${timeAgo}`,
    ];
  }

  // Determine the tab based on execution status
  const searchParams = new URLSearchParams();
  const isReview = execution.status === AgentExecutionStatus.REVIEW;
  searchParams.set("activeTab", isReview ? "reviews" : "runs");
  searchParams.set("activeItem", execution.id);
  const linkUrl = `/library/agents/${execution.library_agent_id}?${searchParams.toString()}`;
  const withExecutionLink = execution.library_agent_id && execution.id;

  const content = (
    <>
      {/* Icon + Agent Name */}
      <div className="flex items-center space-x-2">
        {getStatusIcon()}
        <Text variant="body-medium" className="max-w-44 truncate text-gray-900">
          {execution.agent_name}
        </Text>
      </div>

      {/* Agent Message - Indented */}
      <div className="ml-7 pt-1">
        {getItemDisplay().map((line, index) => (
          <Text
            key={index}
            variant="small"
            className={index === 0 ? "!text-zinc-600" : "!text-zinc-500"}
          >
            {line}
          </Text>
        ))}
      </div>
    </>
  );

  return withExecutionLink ? (
    <Link
      className="block cursor-pointer border-b border-slate-50 px-2 py-3 transition-colors last:border-b-0 hover:bg-bgLightGrey"
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
