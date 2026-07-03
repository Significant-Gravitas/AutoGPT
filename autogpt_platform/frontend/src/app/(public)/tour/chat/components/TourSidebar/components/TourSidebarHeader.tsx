"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { SidebarHeader, useSidebar } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { SidebarSimpleIcon } from "@phosphor-icons/react";
import Link from "next/link";

// Mirror of AppSidebarHeader, with the logo pointing back at the tour
// instead of the (auth-gated) copilot home.
export function TourSidebarHeader() {
  const { state, toggleSidebar } = useSidebar();
  const isCollapsed = state === "collapsed";

  return (
    <SidebarHeader className="flex animate-fade-in flex-row items-center justify-between gap-2 p-2 group-data-[collapsible=icon]:flex-col">
      <Link
        href="/tour"
        aria-label="AutoGPT"
        className={cn(
          "flex items-center",
          isCollapsed && "group-focus-within:hidden group-hover:hidden",
        )}
      >
        {isCollapsed ? (
          <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-8" />
        ) : (
          <AutoGPTLogo className="h-9 w-auto" />
        )}
      </Link>

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
        <SidebarSimpleIcon className="size-5 text-sidebar-foreground" />
      </button>
    </SidebarHeader>
  );
}
