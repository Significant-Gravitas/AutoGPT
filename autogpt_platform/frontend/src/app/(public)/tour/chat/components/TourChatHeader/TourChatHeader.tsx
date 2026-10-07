"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { SidebarTrigger } from "@/components/ui/sidebar";
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
    <header className="flex shrink-0 items-center justify-between gap-2 border-b border-zinc-200/70 bg-white/70 px-3 py-2 backdrop-blur-xs md:px-4">
      <div className="flex min-w-0 items-center gap-1.5">
        {/* On mobile this is the only way to reach the sidebar. */}
        <div className="md:hidden">
          <SidebarTrigger />
        </div>
        <Icon icon={scenarioIcon} className="size-4 shrink-0 text-purple-600" />
        <Text
          variant="body-medium"
          className="truncate bg-linear-to-r from-purple-600 to-purple-400 bg-clip-text text-transparent"
        >
          {scenarioLabel}
        </Text>
      </div>
      <Button
        variant="secondary"
        size="md"
        onClick={handleShare}
        leftIcon={
          isCopied ? (
            <Icon icon={Tick02Icon} className="size-4 text-green-600" />
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
