"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { SidebarHeader, useSidebar } from "@/components/ui/sidebar";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { SidebarSimpleIcon } from "@/components/icons/pika/adapter";
import Link from "next/link";

export function AppSidebarHeader() {
  const { state, toggleSidebar } = useSidebar();
  const isCollapsed = state === "collapsed";

  return (
    <SidebarHeader className="flex animate-fade-in flex-row items-center justify-between gap-2 px-4 pb-4 pt-2 group-data-[collapsible=icon]:flex-col group-data-[collapsible=icon]:px-2">
      <Link
        href="/copilot"
        aria-label="AutoGPT"
        className={cn(
          "flex h-8 items-center",
          isCollapsed && "group-focus-within:hidden group-hover:hidden",
        )}
      >
        {isCollapsed ? (
          <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-6" />
        ) : (
          <AutoGPTLogo className="h-7 w-auto" />
        )}
      </Link>

      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            onClick={toggleSidebar}
            className={cn(
              "size-8 shrink-0 items-center justify-center rounded-md transition-colors hover:bg-zinc-200",
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
    </SidebarHeader>
  );
}
