"use client";

import { useGlobalSearchStore } from "@/app/(platform)/components/GlobalSearchModal/useGlobalSearchStore";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { SidebarHeader, useSidebar } from "@/components/ui/sidebar";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { SidebarSimpleIcon } from "@/components/atoms/AGPTIcon/icons";
import { SearchIcon } from "@/components/icons/SearchIcon";
import Link from "next/link";

const iconButtonClass =
  "size-8 shrink-0 items-center justify-center rounded-md transition-colors hover:bg-zinc-200";

export function AppSidebarHeader() {
  const { state, toggleSidebar } = useSidebar();
  const openSearch = useGlobalSearchStore((store) => store.openSearch);
  const isCollapsed = state === "collapsed";

  return (
    <SidebarHeader className="flex animate-fade-in flex-row items-center justify-between gap-2 px-4 py-2 group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:px-2">
      <Link
        href="/copilot"
        aria-label="AutoGPT"
        className={cn(
          "flex items-center",
          isCollapsed && "group-focus-within:hidden group-hover:hidden",
        )}
      >
        {isCollapsed ? (
          <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-6" />
        ) : (
          <AutoGPTLogo className="h-7 w-auto" />
        )}
      </Link>

      <div className="flex items-center gap-1 group-data-[collapsible=icon]:flex-col">
        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              aria-label="Search"
              onClick={openSearch}
              className={cn(iconButtonClass, isCollapsed ? "hidden" : "flex")}
            >
              <SearchIcon className="size-4 text-sidebar-foreground" />
            </button>
          </TooltipTrigger>
          <TooltipContent side="right">Search</TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <button
              type="button"
              aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
              onClick={toggleSidebar}
              className={cn(
                iconButtonClass,
                isCollapsed
                  ? "hidden group-focus-within:flex group-hover:flex"
                  : "flex",
              )}
            >
              <SidebarSimpleIcon className="size-4 text-sidebar-foreground" />
            </button>
          </TooltipTrigger>
          <TooltipContent side="right">
            {isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          </TooltipContent>
        </Tooltip>
      </div>
    </SidebarHeader>
  );
}
