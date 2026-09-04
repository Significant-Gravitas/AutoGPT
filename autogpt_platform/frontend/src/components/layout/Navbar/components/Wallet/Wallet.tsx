"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import { Text } from "@/components/atoms/Text/Text";
import { TopUpDialog } from "@/components/layout/TopUpPrompt/TopUpDialog/TopUpDialog";
import { cn } from "@/lib/utils";
import { WalletCompactPanel } from "./components/WalletCompactPanel";
import { WalletFullPanel } from "./components/WalletFullPanel";
import { useWallet } from "./useWallet";
import { Wallet01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  compact?: boolean;
}

export function Wallet({ compact = false }: Props) {
  const {
    state,
    groups,
    credits,
    formatCredits,
    flash,
    walletOpen,
    setWalletOpen,
    onWalletOpen,
    walletRef,
    completedCount,
    totalCount,
    topUpOpen,
    onAddCredits,
    onTopUpClose,
  } = useWallet();

  // Do not render until we have both credits and onboarding data
  if (credits === null || !state) return null;

  return (
    <>
      <Popover open={walletOpen} onOpenChange={(open) => setWalletOpen(open)}>
        <PopoverTrigger asChild>
          <div className="relative inline-block">
            <button
              ref={walletRef}
              className={cn(
                "group relative flex flex-nowrap items-center gap-2 rounded-md px-3 py-2 text-sm",
                compact
                  ? "h-8 rounded-lg px-2 py-0 transition-colors hover:bg-zinc-100"
                  : "bg-zinc-50",
              )}
              onClick={onWalletOpen}
            >
              <Icon
                icon={Wallet01Icon}
                size={20}
                className={cn("inline-block", !compact && "xl:hidden")}
              />
              <div>
                {!compact && (
                  <span className="mr-1 hidden xl:inline-block">
                    Earn credits{" "}
                  </span>
                )}
                <span
                  className={cn(
                    compact ? "text-xs font-medium" : "text-sm font-semibold",
                  )}
                >
                  {formatCredits(credits)}
                </span>
                {!compact &&
                  completedCount !== null &&
                  completedCount < totalCount && (
                    <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-violet-600"></span>
                  )}
                {!compact && (
                  <div className="absolute bottom-[-2.5rem] left-1/2 z-50 hidden -translate-x-1/2 transform whitespace-nowrap rounded-small bg-white px-4 py-2 shadow-md group-hover:block">
                    <Text variant="body-medium">
                      {completedCount} of {totalCount} rewards claimed
                    </Text>
                  </div>
                )}
              </div>
            </button>
            <div
              className={cn(
                "pointer-events-none absolute inset-0 bg-violet-400 duration-2000 ease-in-out",
                compact ? "rounded-lg" : "rounded-md",
                flash ? "opacity-50 duration-0" : "opacity-0",
              )}
            />
          </div>
        </PopoverTrigger>
        <PopoverContent
          side={compact ? "top" : "bottom"}
          align={compact ? "start" : "end"}
          collisionPadding={16}
          className={cn(
            "z-50",
            compact
              ? "w-[22rem] rounded-2xlarge p-2"
              : "relative -top-12 w-[28.5rem] px-4 py-4",
          )}
        >
          {compact ? (
            <WalletCompactPanel
              groups={groups}
              completedSteps={state.completedSteps}
              formattedCredits={formatCredits(credits)}
              onAddCredits={onAddCredits}
            />
          ) : (
            <WalletFullPanel
              groups={groups}
              formattedCredits={formatCredits(credits)}
            />
          )}
        </PopoverContent>
      </Popover>
      {compact && (
        <TopUpDialog
          isOpen={topUpOpen}
          onClose={onTopUpClose}
          variant="add-credits"
        />
      )}
    </>
  );
}
