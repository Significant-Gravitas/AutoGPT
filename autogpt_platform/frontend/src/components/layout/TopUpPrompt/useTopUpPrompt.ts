import { createContext, useContext } from "react";

interface TopUpPromptContextValue {
  isOutOfCredits: boolean;
  openTopUp: () => void;
  closeTopUp: () => void;
  /** The "what would make topping up worth it" card is on screen. */
  isTopUpCardOpen: boolean;
}

export const TopUpPromptContext = createContext<TopUpPromptContextValue | null>(
  null,
);

// The prompt is an optional enhancement, so consumers rendered without a
// provider (e.g. a page tested outside the platform layout) fall back to an
// inert value rather than crashing the host page.
const inertPrompt: TopUpPromptContextValue = {
  isOutOfCredits: false,
  openTopUp: () => {},
  closeTopUp: () => {},
  isTopUpCardOpen: false,
};

export function useTopUpPrompt() {
  return useContext(TopUpPromptContext) ?? inertPrompt;
}
