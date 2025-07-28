"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Bell } from "@phosphor-icons/react";
import { AgentExecutionWithInfo, EXECUTION_DISPLAY_LIMIT } from "../../helpers";
import { ActivityItem } from "../ActivityItem";
import { getSortedExecutions } from "./helpers";

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

  const sortedExecutions = getSortedExecutions({
    activeExecutions: displayActiveExecutions,
    recentCompletions: displayRecentCompletions,
    recentFailures: displayRecentFailures,
  });

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
