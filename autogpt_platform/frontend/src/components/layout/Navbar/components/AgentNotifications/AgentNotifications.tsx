"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Bell } from "@phosphor-icons/react";
import { useState } from "react";
import { NotificationDropdown } from "./components/NotificationDropdown";
import { formatNotificationCount } from "./helpers";
import { useAgentNotifications } from "./useAgentNotifications";

export function AgentNotifications() {
  const [isOpen, setIsOpen] = useState(false);
  const { activeExecutions, recentCompletions, recentFailures } =
    useAgentNotifications();

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <button
          className={`relative rounded-full p-2 transition-colors hover:bg-white ${isOpen ? "bg-white" : ""}`}
          title="Agent Activity"
        >
          <Bell size={22} className="text-black" />

          {/* Running Agents Badge */}
          {activeExecutions.length > 0 && (
            <div className="absolute right-[1px] top-[0.5px] flex h-5 w-5 items-center justify-center rounded-full bg-purple-600 text-[10px] font-medium text-white">
              {formatNotificationCount(activeExecutions.length)}

              {/* Rotating Spinner */}
              <div className="absolute -inset-0.5 animate-spin rounded-full border-[3px] border-transparent border-r-purple-200 border-t-purple-200" />
            </div>
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
