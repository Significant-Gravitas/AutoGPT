"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { SidebarHeader, useSidebar } from "@/components/ui/sidebar";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { SidebarLeftIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { SidebarSearch } from "../SidebarSearch/SidebarSearch";

export function AppSidebarHeader() {
  const { state, toggleSidebar } = useSidebar();
  const isCollapsed = state === "collapsed";

  const toggleButton = (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          onClick={toggleSidebar}
          className={cn(
            "size-8 shrink-0 items-center justify-center rounded-md transition-colors hover:bg-zinc-200",
            isCollapsed
              ? "absolute inset-0 hidden group-focus-within:flex group-hover:flex"
              : "flex",
          )}
        >
          <Icon
            icon={SidebarLeftIcon}
            className="size-4 text-sidebar-foreground/90 group-data-[collapsible=icon]:size-4.5"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent side="right">
        {isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
      </TooltipContent>
    </Tooltip>
  );

  return (
    <SidebarHeader className="mb-1 flex animate-fade-in flex-row items-center justify-between gap-2 p-2 group-data-[collapsible=icon]:flex-col">
      <div
        className={cn(
          "relative flex items-center",
          isCollapsed && "size-8 shrink-0",
        )}
      >
        <Link
          href="/copilot"
          aria-label="AutoGPT"
          className={cn(
            "flex items-center",
            isCollapsed &&
              "size-8 items-center justify-center group-focus-within:hidden group-hover:hidden",
          )}
        >
          {isCollapsed ? (
            <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-7" />
          ) : (
            <AutoGPTLogo className="-mt-1 ml-2.5 h-7 w-auto" />
          )}
        </Link>
        {isCollapsed ? toggleButton : null}
      </div>

      <div className="flex shrink-0 items-center gap-1 group-data-[collapsible=icon]:flex-col">
        <SidebarSearch />
        {isCollapsed ? null : toggleButton}
      </div>
    </SidebarHeader>
  );
}
