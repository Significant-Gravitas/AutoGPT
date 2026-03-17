"use client";

import React, { createContext, useContext } from "react";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { RunVariant } from "./useAgentRunModal";

export interface RunAgentModalContextValue {
  agent: LibraryAgent;
  defaultRunType: RunVariant;
  // Preset / Trigger
  presetName: string;
  setPresetName: (value: string) => void;
  presetDescription: string;
  setPresetDescription: (value: string) => void;
  // Inputs
  inputValues: Record<string, any>;
  setInputValue: (key: string, value: any) => void;
  agentInputFields: Record<string, any>;
  // Credentials
  inputCredentials: Record<string, any>;
  setInputCredentialsValue: (key: string, value: any | undefined) => void;
  agentCredentialsInputFields: Record<string, any>;
}

const RunAgentModalContext = createContext<RunAgentModalContextValue | null>(
  null,
);

export function useRunAgentModalContext(): RunAgentModalContextValue {
  const ctx = useContext(RunAgentModalContext);
  if (!ctx) throw new Error("RunAgentModalContext missing provider");
  return ctx;
}

interface ProviderProps {
  value: RunAgentModalContextValue;
  children: React.ReactNode;
}

export function RunAgentModalContextProvider({
  value,
  children,
}: ProviderProps) {
  return (
    <RunAgentModalContext.Provider value={value}>
      {children}
    </RunAgentModalContext.Provider>
  );
}
