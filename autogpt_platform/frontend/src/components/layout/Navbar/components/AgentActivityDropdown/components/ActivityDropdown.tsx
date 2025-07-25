"use client";

import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { Text } from "@/components/atoms/Text/Text";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Bell } from "@phosphor-icons/react";
import { AgentExecutionWithInfo, EXECUTION_DISPLAY_LIMIT } from "../helpers";
import { ActivityItem } from "./ActivityItem";

interface Props {
  activeExecutions: AgentExecutionWithInfo[];
  recentCompletions: AgentExecutionWithInfo[];
  recentFailures: AgentExecutionWithInfo[];
}

export function ActivityDropdown({
  activeExecutions,
  recentCompletions,
  recentFailures,
}: Props) {
  // Limit arrays for display purposes
  const displayActiveExecutions = activeExecutions.slice(
    0,
    EXECUTION_DISPLAY_LIMIT,
  );
  const displayRecentCompletions = recentCompletions.slice(
    0,
    EXECUTION_DISPLAY_LIMIT,
  );
  const displayRecentFailures = recentFailures.slice(
    0,
    EXECUTION_DISPLAY_LIMIT,
  );

  // Combine and sort all executions - running/queued at top, then by most recent
  function getSortedExecutions() {
    const allExecutions = [
      ...displayActiveExecutions.map((e) => ({
        ...e,
        type: "running" as const,
      })),
      ...displayRecentCompletions.map((e) => ({
        ...e,
        type: "completed" as const,
      })),
      ...displayRecentFailures.map((e) => ({ ...e, type: "failed" as const })),
    ];

    return allExecutions
      .sort((a, b) => {
        // Priority order: RUNNING > QUEUED > everything else
        const aIsRunning = a.status === AgentExecutionStatus.RUNNING;
        const bIsRunning = b.status === AgentExecutionStatus.RUNNING;
        const aIsQueued = a.status === AgentExecutionStatus.QUEUED;
        const bIsQueued = b.status === AgentExecutionStatus.QUEUED;

        // RUNNING agents always at the very top
        if (aIsRunning && !bIsRunning) return -1;
        if (!aIsRunning && bIsRunning) return 1;

        // If both are running, sort by most recent start time
        if (aIsRunning && bIsRunning) {
          if (!a.started_at || !b.started_at) return 0;
          return (
            new Date(b.started_at).getTime() - new Date(a.started_at).getTime()
          );
        }

        // QUEUED agents come second
        if (aIsQueued && !bIsQueued) return -1;
        if (!aIsQueued && bIsQueued) return 1;

        // If both are queued, sort by most recent start time
        if (aIsQueued && bIsQueued) {
          if (!a.started_at || !b.started_at) return 0;
          return (
            new Date(b.started_at).getTime() - new Date(a.started_at).getTime()
          );
        }

        // For everything else (completed, failed, etc.), sort by most recent end time
        const aTime = a.ended_at;
        const bTime = b.ended_at;

        if (!aTime || !bTime) return 0;
        return new Date(bTime).getTime() - new Date(aTime).getTime();
      })
      .slice(0, EXECUTION_DISPLAY_LIMIT);
  }

  const sortedExecutions = getSortedExecutions();

  return (
    <div>
      {/* Header */}
      <div className="sticky top-0 z-10 px-4 pb-1 pt-4">
        <Text variant="large-semibold" className="!text-black">
          Agent Activity
        </Text>
      </div>

      {/* Content */}
      <ScrollArea
        className="min-h-[10rem]"
        data-testid="agent-activity-dropdown"
      >
        {sortedExecutions.length > 0 ? (
          <div className="p-2">
            {sortedExecutions.map((execution) => (
              <ActivityItem key={execution.id} execution={execution} />
            ))}
          </div>
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-5 pb-8 pt-6">
            <div className="mx-auto inline-flex flex-col items-center justify-center rounded-full bg-lightGrey p-6">
              <Bell className="h-6 w-6 text-zinc-300" />
            </div>
            <div className="flex flex-col items-center justify-center">
              <Text variant="body-medium" className="!text-black">
                Nothing to show yet
              </Text>
              <Text variant="body" className="!text-zinc-500">
                Start an agent to get updates
              </Text>
            </div>
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
