"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import {
  SidebarHeader,
  useSidebar,
} from "@/components/molecules/Sidebar/Sidebar";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipPortal,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { cn } from "@/lib/utils";
import Link from "next/link";
import { SidebarLeftIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { SidebarSearch } from "../SidebarSearch/SidebarSearch";

export function AppSidebarHeader() {
  const { state, toggleSidebar } = useSidebar();
  const isCollapsed = state === "collapsed";

  const toggleButton = (
    <Tooltip inheritDelay>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          onClick={toggleSidebar}
          className={cn(
            "size-8 shrink-0 items-center justify-center rounded-lg text-sidebar-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground max-md:size-10",
            isCollapsed
              ? "absolute inset-0 hidden group-focus-within:flex group-hover:flex"
              : "flex",
          )}
        >
          <Icon
            icon={SidebarLeftIcon}
            tone="inherit"
            className="size-4 max-md:size-5"
          />
        </button>
      </TooltipTrigger>
      <TooltipPortal>
        <TooltipContent side="right">
          {isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        </TooltipContent>
      </TooltipPortal>
    </Tooltip>
  );

  return (
    <SidebarHeader
      size="shell"
      className="sticky top-0 z-10 flex shrink-0 animate-fade-in flex-col bg-sidebar"
    >
      <div className="flex items-center justify-between gap-2 group-data-[collapsible=icon]:flex-col">
        <div
          className={cn(
            "relative flex items-center",
            isCollapsed && "size-8 shrink-0",
          )}
        >
          <Link
            href="/home"
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
              <AutoGPTLogo className="-mt-1 ml-2.5 h-7 w-auto max-md:mt-2 max-md:h-9" />
            )}
          </Link>
          {isCollapsed ? toggleButton : null}
        </div>

        <div className="flex shrink-0 items-center gap-1 group-data-[collapsible=icon]:flex-col">
          <SidebarSearch />
          {isCollapsed ? null : toggleButton}
        </div>
      </div>
    </SidebarHeader>
  );
}
