"use client";

import { useGlobalSearchStore } from "@/app/(platform)/components/GlobalSearchModal/useGlobalSearchStore";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Tooltip,
  TooltipContent,
  TooltipPortal,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Search01Icon } from "@hugeicons/core-free-icons";

export function SidebarSearch() {
  const openSearch = useGlobalSearchStore((state) => state.openSearch);

  return (
    <Tooltip>
      <TooltipTrigger
        render={
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            aria-label="Search"
            withTooltip={false}
            onClick={openSearch}
            className="shrink-0 rounded-md hover:border-transparent hover:bg-zinc-200"
          >
            <Icon
              icon={Search01Icon}
              className="size-4 text-sidebar-foreground/90 group-data-[collapsible=icon]:size-4.5"
            />
          </Button>
        }
      />
      <TooltipPortal>
        <TooltipContent side="right">Search</TooltipContent>
      </TooltipPortal>
    </Tooltip>
  );
}
