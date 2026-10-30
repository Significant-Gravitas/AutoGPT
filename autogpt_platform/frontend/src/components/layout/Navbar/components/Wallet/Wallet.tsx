"use client";

import {
  Popover,
  PopoverContent,
} from "@/components/molecules/Popover/Popover";
import { TopUpDialog } from "@/components/layout/TopUpPrompt/TopUpDialog/TopUpDialog";
import { useIsBillingEnabled } from "@/hooks/useIsBillingEnabled";
import { cn } from "@/lib/utils";
import { UsageCreditsDialog } from "@/components/organisms/UsageCreditsDialog/UsageCreditsDialog";
import { useId } from "react";
import { WalletCompactPanel } from "./components/WalletCompactPanel";
import { WalletFullPanel } from "./components/WalletFullPanel";
import { useWallet } from "./useWallet";
import { WalletTrigger } from "./components/WalletTrigger";

interface Props {
  compact?: boolean;
}

export function Wallet({ compact = false }: Props) {
  const isBillingEnabled = useIsBillingEnabled();
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
    explainerOpen,
    onOpenExplainer,
    onExplainerOpenChange,
  } = useWallet();
  const trialNoteID = useId();

  // Do not render until we have both credits and onboarding data
  if (!isBillingEnabled || credits === null || !state) return null;

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
          unseen={!state.walletShown}
        />
        <PopoverContent
          side={compact ? "top" : "bottom"}
          align={compact ? "start" : "end"}
          collisionPadding={16}
          onCloseAutoFocus={(event) => {
            if (topUpOpen || explainerOpen) event.preventDefault();
          }}
          aria-label={compact ? "Usage and credits" : "Automation credits"}
          aria-describedby={compact ? trialNoteID : undefined}
          className={cn(
            "z-50",
            compact
              ? "max-h-[var(--radix-popover-content-available-height)] w-[22rem] max-w-[calc(100vw-2rem)] overflow-y-auto overscroll-contain !rounded-2xlarge p-0 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-surface-active"
              : "relative -top-12 w-[28.5rem] px-4 py-4",
          )}
        >
          {compact ? (
            <WalletCompactPanel
              groups={groups}
              completedSteps={state.completedSteps}
              formattedCredits={formatCredits(credits)}
              trialNoteID={trialNoteID}
              onAddCredits={onAddCredits}
              onOpenExplainer={onOpenExplainer}
              onNavigate={() => setWalletOpen(false)}
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
        <>
          <TopUpDialog
            isOpen={topUpOpen}
            onClose={onTopUpClose}
            variant="add-credits"
          />
          <UsageCreditsDialog
            isOpen={explainerOpen}
            onOpenChange={onExplainerOpenChange}
          />
        </>
      )}
    </>
  );
}
