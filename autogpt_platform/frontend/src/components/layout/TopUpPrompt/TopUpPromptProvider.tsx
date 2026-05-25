"use client";

import { ReactNode, useState } from "react";

import useCredits from "@/hooks/useCredits";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

import { DailyTopUpAutoOpener } from "./DailyTopUpAutoOpener";
import { TopUpDialog } from "./TopUpDialog/TopUpDialog";
import { TopUpPromptContext } from "./useTopUpPrompt";

interface Props {
  children: ReactNode;
}

export function TopUpPromptProvider({ children }: Props) {
  const isBillingEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const { credits, autoTopUpConfig } = useCredits({
    fetchInitialCredits: true,
    fetchInitialAutoTopUpConfig: true,
  });

  // Backend treats amount === 0 as disabled, so auto-refill only actually refills when amount > 0; suppress the nudge only then.
  const autoRefillEnabled = !!autoTopUpConfig && autoTopUpConfig.amount > 0;
  // Both fetches resolve independently; wait for the auto-top-up config too,
  // otherwise a user with auto-refill enabled briefly looks out-of-credits
  // (config still null) and the daily modal fires spuriously.
  const isOutOfCredits =
    !!isBillingEnabled &&
    credits !== null &&
    credits <= 0 &&
    autoTopUpConfig !== null &&
    !autoRefillEnabled;

  const [isOpen, setIsOpen] = useState(false);

  function openTopUp() {
    setIsOpen(true);
  }

  return (
    <TopUpPromptContext.Provider value={{ isOutOfCredits, openTopUp }}>
      {children}
      <TopUpDialog isOpen={isOpen} onClose={() => setIsOpen(false)} />
      {isOutOfCredits && <DailyTopUpAutoOpener />}
    </TopUpPromptContext.Provider>
  );
}
