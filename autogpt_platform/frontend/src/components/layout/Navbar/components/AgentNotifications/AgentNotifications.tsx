"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Bell } from "@phosphor-icons/react";
import { NotificationDropdown } from "./components/NotificationDropdown";
import { formatNotificationCount } from "./helpers";
import { useAgentNotifications } from "./useAgentNotifications";

export function AgentNotifications() {
  const { activeExecutions, recentCompletions, recentFailures, isConnected } =
    useAgentNotifications();

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          className="relative rounded-lg p-2 transition-colors hover:bg-white/10"
          title="Agent Activity"
        >
          <Bell size={22} className="text-black" />

          {/* Running Agents Badge */}
          {activeExecutions.length > 0 && (
            <div className="absolute -right-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full bg-blue-500 text-xs font-medium text-white">
              {formatNotificationCount(activeExecutions.length)}

              {/* Rotating Spinner */}
              <div className="absolute -inset-1 animate-spin rounded-full border-2 border-transparent border-t-blue-300" />
            </div>
          )}

          {/* Connection Status Indicator - only show when no running agents */}
          {!isConnected && activeExecutions.length === 0 && (
            <div className="absolute -right-1 -top-1 h-2 w-2 rounded-full bg-yellow-500" />
          )}
        </button>
      </PopoverTrigger>

      <PopoverContent className="w-80 p-0" align="center" sideOffset={8}>
        <NotificationDropdown
          activeExecutions={activeExecutions}
          recentCompletions={recentCompletions}
          recentFailures={recentFailures}
        />
      </PopoverContent>
    </Popover>
  );
}
