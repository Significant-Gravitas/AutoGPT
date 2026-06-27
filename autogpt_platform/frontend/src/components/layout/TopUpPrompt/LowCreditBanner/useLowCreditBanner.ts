import { useState } from "react";

import { Key } from "@/services/storage/local-storage";

import { markShownToday, wasShownToday } from "../helpers";
import { useTopUpPrompt } from "../useTopUpPrompt";

export function useLowCreditBanner() {
  const { isOutOfCredits, openTopUp } = useTopUpPrompt();
  const [dismissed, setDismissed] = useState(() =>
    wasShownToday(Key.LOW_CREDIT_BANNER_DISMISSED),
  );

  function dismiss() {
    markShownToday(Key.LOW_CREDIT_BANNER_DISMISSED);
    setDismissed(true);
  }

  return { visible: isOutOfCredits && !dismissed, openTopUp, dismiss };
}
