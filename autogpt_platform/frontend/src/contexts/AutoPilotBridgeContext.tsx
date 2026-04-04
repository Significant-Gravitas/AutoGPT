"use client";

import { createContext, useContext, useCallback, useState } from "react";
import { useRouter } from "next/navigation";

const STORAGE_KEY = "autopilot_pending_prompt";

interface AutoPilotBridgeState {
  /** Pending prompt to be injected into AutoPilot chat. */
  pendingPrompt: string | null;
  /** Queue a prompt that the Home/Copilot tab will pick up. */
  sendPrompt: (prompt: string) => void;
  /** Consume and clear the pending prompt (called by the chat page). */
  consumePrompt: () => string | null;
}

const AutoPilotBridgeContext = createContext<AutoPilotBridgeState | null>(null);

interface Props {
  children: React.ReactNode;
}

export function AutoPilotBridgeProvider({ children }: Props) {
  const router = useRouter();

  // Hydrate from sessionStorage in case we just navigated here
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(() => {
    if (typeof window === "undefined") return null;
    return sessionStorage.getItem(STORAGE_KEY);
  });

  const sendPrompt = useCallback(
    (prompt: string) => {
      // Persist to sessionStorage so it survives client-side navigation
      sessionStorage.setItem(STORAGE_KEY, prompt);
      setPendingPrompt(prompt);
      // Use Next.js router for client-side navigation (preserves React tree)
      router.push("/");
    },
    [router],
  );

  const consumePrompt = useCallback((): string | null => {
    const prompt = pendingPrompt ?? sessionStorage.getItem(STORAGE_KEY);
    if (prompt !== null) {
      sessionStorage.removeItem(STORAGE_KEY);
      setPendingPrompt(null);
    }
    return prompt;
  }, [pendingPrompt]);

  return (
    <AutoPilotBridgeContext.Provider
      value={{ pendingPrompt, sendPrompt, consumePrompt }}
    >
      {children}
    </AutoPilotBridgeContext.Provider>
  );
}

export function useAutoPilotBridge(): AutoPilotBridgeState {
  const context = useContext(AutoPilotBridgeContext);
  if (!context) {
    // Return a no-op implementation when used outside the provider
    // (e.g. in tests or isolated component renders).
    return {
      pendingPrompt: null,
      sendPrompt: () => {},
      consumePrompt: () => null,
    };
  }
  return context;
}
