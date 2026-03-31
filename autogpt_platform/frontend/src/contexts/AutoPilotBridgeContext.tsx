"use client";

import { createContext, useContext, useState } from "react";

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
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);

  function sendPrompt(prompt: string) {
    setPendingPrompt(prompt);
    // Navigate to the Home / Copilot tab.
    // Using window.location is the simplest approach that works across the
    // Next.js app router without coupling to a specific router instance.
    window.location.href = "/";
  }

  function consumePrompt(): string | null {
    const prompt = pendingPrompt;
    setPendingPrompt(null);
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
