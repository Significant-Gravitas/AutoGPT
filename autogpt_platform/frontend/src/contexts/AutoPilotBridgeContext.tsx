"use client";

import { createContext, useContext, useState } from "react";
import { useRouter } from "next/navigation";

const STORAGE_KEY = "autopilot_pending_prompt";

interface AutoPilotBridgeState {
  pendingPrompt: string | null;
  sendPrompt: (prompt: string) => void;
  consumePrompt: () => string | null;
}

const AutoPilotBridgeContext = createContext<AutoPilotBridgeState | null>(null);

interface Props {
  children: React.ReactNode;
}

export function AutoPilotBridgeProvider({ children }: Props) {
  const router = useRouter();

  const [pendingPrompt, setPendingPrompt] = useState<string | null>(() => {
    if (typeof window === "undefined") return null;
    return sessionStorage.getItem(STORAGE_KEY);
  });

  function sendPrompt(prompt: string) {
    sessionStorage.setItem(STORAGE_KEY, prompt);
    setPendingPrompt(prompt);
    router.push("/");
  }

  function consumePrompt(): string | null {
    const prompt = pendingPrompt ?? sessionStorage.getItem(STORAGE_KEY);
    if (prompt !== null) {
      sessionStorage.removeItem(STORAGE_KEY);
      setPendingPrompt(null);
    }
    return prompt;
  }

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
