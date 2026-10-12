"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { PopoverTrigger } from "@/components/molecules/Popover/Popover";
import { cn } from "@/lib/utils";
import { Wallet01Icon } from "@hugeicons/core-free-icons";
import { Ref, useState } from "react";

interface Props {
  compact: boolean;
  open: boolean;
  tooltipDisabled: boolean;
  walletRef: Ref<HTMLButtonElement>;
  onClick: () => void;
  formattedCredits: string;
  completedCount: number | null;
  totalCount: number;
  flash: boolean;
}

export function WalletTrigger({
  compact,
  open,
  tooltipDisabled,
  walletRef,
  onClick,
  formattedCredits,
  completedCount,
  totalCount,
  flash,
}: Props) {
  const [tooltipOpen, setTooltipOpen] = useState(false);

  return (
    <Tooltip
      open={compact && !open && !tooltipDisabled && tooltipOpen}
      onOpenChange={setTooltipOpen}
    >
      <div className="relative inline-block">
        <TooltipTrigger asChild>
          <PopoverTrigger asChild>
            <button
              ref={walletRef}
              type="button"
              aria-label={compact ? "Usage and credits" : undefined}
              data-state={open ? "open" : "closed"}
              onClick={onClick}
              className={cn(
                "group relative flex flex-nowrap items-center gap-2 rounded-md px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-zinc-900 focus-visible:ring-offset-2",
                compact
                  ? "size-8 justify-center rounded-lg p-0 text-zinc-700 transition-colors data-[state=open]:bg-zinc-100 data-[state=open]:text-zinc-900 hover:bg-zinc-100"
                  : "bg-zinc-50",
              )}
            >
              <Icon
                icon={Wallet01Icon}
                size={20}
                aria-hidden
                className={cn("inline-block", !compact && "xl:hidden")}
              />
              {!compact && (
                <ClassicBalance
                  formattedCredits={formattedCredits}
                  completedCount={completedCount}
                  totalCount={totalCount}
                />
              )}
            </button>
          </PopoverTrigger>
        </TooltipTrigger>
        <div
          aria-hidden
          className={cn(
            "pointer-events-none absolute inset-0 bg-violet-400 duration-2000 ease-in-out",
            compact ? "rounded-lg" : "rounded-md",
            flash ? "opacity-50 duration-0" : "opacity-0",
          )}
        />
      </div>
      {compact && <TooltipContent side="top">Usage and credits</TooltipContent>}
    </Tooltip>
  );
}

function ClassicBalance({
  formattedCredits,
  completedCount,
  totalCount,
}: Pick<Props, "formattedCredits" | "completedCount" | "totalCount">) {
  return (
    <div>
      <span className="mr-1 hidden xl:inline-block">Earn credits </span>
      <span className="text-sm font-semibold">{formattedCredits}</span>
      {completedCount !== null && completedCount < totalCount && (
        <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-violet-600" />
      )}
      <div className="absolute bottom-[-2.5rem] left-1/2 z-50 hidden -translate-x-1/2 transform whitespace-nowrap rounded-small bg-white px-4 py-2 shadow-md group-hover:block">
        <Text variant="body-medium">
          {completedCount} of {totalCount} rewards claimed
        </Text>
      </div>
    </div>
  );
}
