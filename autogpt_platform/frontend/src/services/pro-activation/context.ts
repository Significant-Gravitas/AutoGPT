import { createContext } from "react";

export const ProActivationContext = createContext<{
  start: (returnTo?: string) => Promise<void>;
  isBusy: boolean;
  isReady: boolean;
} | null>(null);
