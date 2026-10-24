"use client";

import { useGlobalSearchStore } from "@/app/(platform)/components/GlobalSearchModal/useGlobalSearchStore";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipPortal,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Search01Icon } from "@hugeicons/core-free-icons";

export function SidebarSearch() {
  const openSearch = useGlobalSearchStore((state) => state.openSearch);

  return (
    <Tooltip inheritDelay>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label="Search"
          onClick={openSearch}
          className="flex size-8 shrink-0 items-center justify-center rounded-md transition-colors hover:bg-surface-active max-md:size-10"
        >
          <Icon
            icon={Search01Icon}
            className="size-4 text-sidebar-foreground/90 group-data-[collapsible=icon]:size-4.5 max-md:size-5"
          />
        </button>
      </TooltipTrigger>
      <TooltipPortal>
        <TooltipContent side="right">Search</TooltipContent>
      </TooltipPortal>
    </Tooltip>
  );
}
