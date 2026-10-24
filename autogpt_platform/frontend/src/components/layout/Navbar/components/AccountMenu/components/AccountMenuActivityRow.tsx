"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import * as React from "react";
import { ActivityDropdown } from "../../AgentActivityDropdown/components/ActivityDropdown/ActivityDropdown";
import { formatNotificationCount } from "../../AgentActivityDropdown/helpers";
import { useAgentActivityDropdown } from "../../AgentActivityDropdown/useAgentActivityDropdown";
import { ArrowRight01Icon, Pulse01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

const rowClasses =
  // TODO(tokens): no exact role for text-neutral-700 (#404040), hover:bg-neutral-100 (#F5F5F5), focus-visible:bg-neutral-100 (#F5F5F5), data-[state=open]:bg-neutral-100 (#F5F5F5)
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
            // TODO(tokens): no exact role for bg-neutral-900 (#171717)
            className="absolute left-0 top-1/2 h-5 w-[3px] -translate-y-1/2 rounded-full bg-neutral-900 opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100 group-data-[state=open]:opacity-100"
            aria-hidden="true"
          />
          <span className="relative z-10 flex shrink-0 items-center">
            <Icon icon={Pulse01Icon} className="h-[18px] w-[18px] shrink-0" />
          </span>
          <span className="relative z-10 truncate">Activity</span>
          {totalCount > 0 && (
            <span
              data-testid="account-menu-activity-total"
              // TODO(tokens): no exact role for bg-neutral-200 (#E5E5E5), text-neutral-700 (#404040)
              className="relative z-10 flex h-5 min-w-5 items-center justify-center rounded-full bg-neutral-200 px-1.5 text-caption-sm font-semibold leading-inherit text-neutral-700"
            >
              {formatNotificationCount(totalCount)}
            </span>
          )}
          <span className="flex-1" aria-hidden="true" />
          <Icon
            icon={ArrowRight01Icon}
            // TODO(tokens): no exact role for text-neutral-700 (#404040)
            className="relative z-10 shrink-0 text-neutral-700"
            size={16}
            aria-hidden="true"
          />
        </button>
      </PopoverTrigger>

      <PopoverContent
        id={popupId}
        side="right"
        align="end"
        sideOffset={12}
        // TODO(tokens): no exact role for border-neutral-200 (#E5E5E5)
        className="w-80 rounded-2xl border border-neutral-200 p-0 shadow-lg"
        data-testid="account-menu-activity-popover"
      >
        <ActivityDropdown
          activeExecutions={activeExecutions}
          recentCompletions={recentCompletions}
          recentFailures={recentFailures}
        />
      </PopoverContent>
    </Popover>
  );
}
