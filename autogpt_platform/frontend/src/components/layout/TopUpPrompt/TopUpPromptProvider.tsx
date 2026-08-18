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

  // Backend treats amount === 0 as disabled, so auto-refill only actually
  // refills when amount > 0; suppress the nudge only then.
  const autoRefillEnabled = !!autoTopUpConfig && autoTopUpConfig.amount > 0;
  // Requiring `autoTopUpConfig !== null` serves two purposes. Both fetches
  // resolve independently, so it avoids a user with auto-refill enabled briefly
  // looking out-of-credits (config still null) and firing the daily modal
  // spuriously. It is also fail-closed by design: if GET /credits/auto-top-up
  // errors, autoTopUpConfig stays null and the prompt stays hidden, rather than
  // risk nudging a user who actually has auto-refill enabled.
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

  function closeTopUp() {
    setIsOpen(false);
  }

  return (
    <TopUpPromptContext.Provider
      value={{ isOutOfCredits, openTopUp, closeTopUp }}
    >
      {children}
      <TopUpDialog isOpen={isOpen} onClose={closeTopUp} />
      {isOutOfCredits && <DailyTopUpAutoOpener />}
    </TopUpPromptContext.Provider>
  );
}
