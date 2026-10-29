"use client";

import { Button } from "@/components/atoms/Button/Button";
import { SidebarTrigger } from "@/components/molecules/Sidebar/Sidebar";
import { useTourChatHeader } from "./useTourChatHeader";
import { Link02Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  scenarioLabel: string;
  scenarioIcon: IconSvgElement;
}

export function TourChatHeader({ scenarioLabel, scenarioIcon }: Props) {
  const { isCopied, handleShare } = useTourChatHeader();

  return (
    <header className="flex shrink-0 items-center justify-between gap-2 border-b border-border-base/70 bg-surface/70 px-3 py-2 backdrop-blur-sm md:px-4">
      <div className="flex min-w-0 items-center gap-1.5">
        {/* On mobile this is the only way to reach the sidebar. */}
        <div className="md:hidden">
          <SidebarTrigger />
        </div>
        <span className="contents text-brand">
          <Icon icon={scenarioIcon} className="size-4 shrink-0" />
        </span>
        {/* oxlint-disable-next-line design-system/no-raw-colors -- the scenario title's purple-to-indigo gradient, no indigo role */}
        <p className="truncate bg-gradient-to-r from-brand-hover to-indigo-500 bg-clip-text font-sans text-body font-medium text-transparent">
          {scenarioLabel}
        </p>
      </div>
      <Button
        variant="secondary"
        size="small"
        onClick={handleShare}
        leftIcon={
          isCopied ? (
            <span className="contents text-success">
              <Icon icon={Tick02Icon} className="size-4" />
            </span>
          ) : (
            <Icon icon={Link02Icon} className="size-4" />
          )
        }
        className="shrink-0"
      >
        <span>{isCopied ? "Link copied" : "Share this demo"}</span>
      </Button>
    </header>
  );
}
