"use client";

import {
  Popover,
  PopoverContent,
} from "@/components/molecules/Popover/Popover";
import { TopUpDialog } from "@/components/layout/TopUpPrompt/TopUpDialog/TopUpDialog";
import { cn } from "@/lib/utils";
import { WalletCompactPanel } from "./components/WalletCompactPanel";
import { WalletFullPanel } from "./components/WalletFullPanel";
import { useWallet } from "./useWallet";
import { WalletTrigger } from "./components/WalletTrigger";

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
        <WalletTrigger
          compact={compact}
          open={walletOpen}
          tooltipDisabled={topUpOpen}
          walletRef={walletRef}
          onClick={onWalletOpen}
          formattedCredits={formatCredits(credits)}
          completedCount={completedCount}
          totalCount={totalCount}
          flash={flash}
        />
        <PopoverContent
          side={compact ? "top" : "bottom"}
          align={compact ? "start" : "end"}
          collisionPadding={16}
          onCloseAutoFocus={(event) => {
            if (topUpOpen) event.preventDefault();
          }}
          aria-label={compact ? "Usage and credits" : "Automation credits"}
          className={cn(
            "z-50",
            compact
              ? "max-h-[var(--radix-popover-content-available-height)] w-[22rem] max-w-[calc(100vw-2rem)] overflow-y-auto overscroll-contain !rounded-2xlarge p-0 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-200"
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
