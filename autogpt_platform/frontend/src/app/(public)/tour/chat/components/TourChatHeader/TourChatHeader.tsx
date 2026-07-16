"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { SidebarTrigger } from "@/components/ui/sidebar";
import type { Icon } from "@phosphor-icons/react";
import { CheckIcon, LinkSimpleIcon } from "@phosphor-icons/react";
import { useTourChatHeader } from "./useTourChatHeader";

interface Props {
  scenarioLabel: string;
  scenarioIcon: Icon;
}

export function TourChatHeader({
  scenarioLabel,
  scenarioIcon: ScenarioIcon,
}: Props) {
  const { isCopied, handleShare } = useTourChatHeader();

  return (
    <header className="flex shrink-0 items-center justify-between gap-2 border-b border-zinc-200/70 bg-white/70 px-3 py-2 backdrop-blur-sm md:px-4">
      <div className="flex min-w-0 items-center gap-1.5">
        {/* SidebarTrigger drops className, so the responsive hiding lives on
            a wrapper — on mobile this is the only way to reach the sidebar. */}
        <div className="md:hidden">
          <SidebarTrigger />
        </div>
        <ScenarioIcon className="size-4 shrink-0 text-violet-600" />
        <Text
          variant="body-medium"
          className="truncate bg-gradient-to-r from-violet-600 to-indigo-500 bg-clip-text text-transparent"
        >
          {scenarioLabel}
        </Text>
      </div>
      <Button
        variant="secondary"
        size="small"
        onClick={handleShare}
        leftIcon={
          isCopied ? (
            <CheckIcon className="size-4 text-emerald-600" weight="bold" />
          ) : (
            <LinkSimpleIcon className="size-4" />
          )
        }
        className="shrink-0"
      >
        {isCopied ? "Link copied" : "Share this demo"}
      </Button>
    </header>
  );
}
