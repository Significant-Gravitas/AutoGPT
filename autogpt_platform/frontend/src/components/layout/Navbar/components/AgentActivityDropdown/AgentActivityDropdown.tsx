"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import { Bell } from "@phosphor-icons/react";
import { ActivityDropdown } from "./components/ActivityDropdown/ActivityDropdown";
import { formatNotificationCount } from "./helpers";
import { useAgentActivityDropdown } from "./useAgentActivityDropdown";

export function AgentActivityDropdown() {
  const {
    activeExecutions,
    recentCompletions,
    recentFailures,
    isOpen,
    setIsOpen,
    isAgentActivityEnabled,
  } = useAgentActivityDropdown();

  const activeCount = activeExecutions.length;

  if (!isAgentActivityEnabled) {
    return null;
  }

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <button
          className={`group relative h-[2.5rem] w-[2.5rem] rounded-full p-2 transition-colors hover:bg-white ${isOpen ? "bg-white" : ""}`}
          data-testid="agent-activity-button"
          aria-label="View Agent Activity"
        >
          <Bell size={22} className="text-black" />

          {activeCount > 0 && (
            <>
              {/* Running Agents Rotating Badge */}
              <div
                data-testid="agent-activity-badge"
                className="absolute right-[1px] top-[0.5px] flex h-5 w-5 items-center justify-center rounded-full bg-purple-600 text-[10px] font-medium text-white"
              >
                {formatNotificationCount(activeCount)}
                <div className="absolute -inset-0.5 animate-spin rounded-full border-[3px] border-transparent border-r-purple-200 border-t-purple-200" />
              </div>
              {/* Running Agent Hover Hint */}
              <div
                data-testid="agent-activity-hover-hint"
                className="absolute bottom-[-2.5rem] left-1/2 z-50 hidden -translate-x-1/2 transform whitespace-nowrap rounded-small bg-white px-4 py-2 shadow-md group-hover:block"
              >
                <Text variant="body-medium">
                  {activeCount} running agent
                  {activeCount > 1 ? "s" : ""}
                </Text>
              </div>
            </>
          )}
        </button>
      </PopoverTrigger>

      <PopoverContent className="w-80 p-0" align="center" sideOffset={8}>
        <ActivityDropdown
          activeExecutions={activeExecutions}
          recentCompletions={recentCompletions}
          recentFailures={recentFailures}
        />
      </PopoverContent>
    </Popover>
  );
}
