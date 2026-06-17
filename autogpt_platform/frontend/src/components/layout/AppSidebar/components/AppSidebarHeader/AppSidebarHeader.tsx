"use client";

import { SidebarHeader, useSidebar } from "@/components/ui/sidebar";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { SidebarSimpleIcon } from "@phosphor-icons/react";
import Image from "next/image";
import Link from "next/link";

export function AppSidebarHeader() {
  const { state, toggleSidebar } = useSidebar();
  const isCollapsed = state === "collapsed";

  return (
    <SidebarHeader className="flex animate-fade-in flex-row items-center justify-between gap-2 p-2 group-data-[collapsible=icon]:flex-col">
      <Link
        href="/copilot"
        aria-label="AutoGPT"
        className={cn("flex items-center", isCollapsed && "group-hover:hidden")}
      >
        {isCollapsed ? (
          <Image
            src="/agpt-logo.png"
            alt="AutoGPT"
            width={545}
            height={553}
            className="size-8"
            priority
          />
        ) : (
          <Image
            src="/autogpt-logo-light-bg.png"
            alt="AutoGPT"
            width={790}
            height={356}
            className="h-7 w-auto"
            priority
          />
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
              isCollapsed ? "hidden group-hover:flex" : "flex",
            )}
          >
            <SidebarSimpleIcon className="size-5 text-sidebar-foreground" />
          </button>
        </TooltipTrigger>
        <TooltipContent side="right">
          {isCollapsed ? "Expand sidebar" : "Collapse sidebar"}
        </TooltipContent>
      </Tooltip>
    </SidebarHeader>
  );
}
