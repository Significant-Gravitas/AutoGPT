"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { CaretRightIcon, PulseIcon } from "@phosphor-icons/react";
import * as React from "react";
import { ActivityDropdown } from "../../AgentActivityDropdown/components/ActivityDropdown/ActivityDropdown";
import { formatNotificationCount } from "../../AgentActivityDropdown/helpers";
import { useAgentActivityDropdown } from "../../AgentActivityDropdown/useAgentActivityDropdown";

const rowClasses =
  "group relative flex w-full items-center gap-3 rounded-lg py-2 pl-3 pr-2 text-left text-sm font-normal text-neutral-700 outline-none transition-colors duration-200 ease-out hover:bg-neutral-100 focus-visible:bg-neutral-100 focus-visible:outline-none data-[state=open]:bg-neutral-100";

export function AccountMenuActivityRow() {
  const { activeExecutions, recentCompletions, recentFailures } =
    useAgentActivityDropdown();

  const popupId = React.useId();

  const totalCount =
    activeExecutions.length + recentCompletions.length + recentFailures.length;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className={rowClasses}
          aria-controls={popupId}
          aria-haspopup="true"
          data-testid="account-menu-activity-trigger"
        >
          <span
            className="absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-full bg-neutral-900 opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100 group-data-[state=open]:opacity-100"
            aria-hidden="true"
          />
          <span className="relative z-10 flex shrink-0 items-center">
            <PulseIcon
              className="h-[18px] w-[18px] shrink-0"
              weight="regular"
            />
          </span>
          <span className="relative z-10 truncate">Activity</span>
          {totalCount > 0 && (
            <span
              data-testid="account-menu-activity-total"
              className="relative z-10 flex h-5 min-w-5 items-center justify-center rounded-full bg-neutral-200 px-1.5 text-[11px] font-semibold text-neutral-700"
            >
              {formatNotificationCount(totalCount)}
            </span>
          )}
          <span className="flex-1" aria-hidden="true" />
          <CaretRightIcon
            className="relative z-10 shrink-0 text-neutral-700"
            size={16}
            weight="regular"
            aria-hidden="true"
          />
        </button>
      </PopoverTrigger>

      <PopoverContent
        id={popupId}
        side="right"
        align="end"
        sideOffset={12}
        className="w-80 rounded-2xl border border-neutral-200 p-0 shadow-lg"
        data-testid="account-menu-activity-popover"
      >
        <ActivityDropdown
          activeExecutions={activeExecutions}
          recentCompletions={recentCompletions}
          recentFailures={recentFailures}
          newLayout
        />
      </PopoverContent>
    </Popover>
  );
}
