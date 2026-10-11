import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import { useEffect } from "react";

import { useCopilotUIStore } from "../../../../store";
import { tierLock } from "./helpers";

// A plan that lapsed with Advanced chosen leaves "advanced" stored, and every
// send is then refused. The server locks Advanced only on a definite "not entitled".
export function useLockedTierFallback(active: AIConnectionOffer | undefined) {
  const { copilotLlmModel, setCopilotLlmModel } = useCopilotUIStore();
  const advancedLocked = Boolean(tierLock(active, "advanced"));
  useEffect(() => {
    if (advancedLocked && copilotLlmModel === "advanced")
      setCopilotLlmModel("standard");
  }, [advancedLocked, copilotLlmModel, setCopilotLlmModel]);
}
