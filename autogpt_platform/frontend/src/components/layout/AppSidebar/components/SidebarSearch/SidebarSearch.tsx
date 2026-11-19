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
          className="flex size-8 shrink-0 items-center justify-center rounded-lg text-shell-foreground transition-colors hover:bg-shell-hover hover:text-shell-foreground-strong max-md:size-10"
        >
          <Icon
            icon={Search01Icon}
            tone="inherit"
            className="size-4 max-md:size-5"
          />
        </button>
      </TooltipTrigger>
      <TooltipPortal>
        <TooltipContent side="right">Search</TooltipContent>
      </TooltipPortal>
    </Tooltip>
  );
}
