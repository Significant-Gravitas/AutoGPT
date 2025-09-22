import { CredentialsMetaInput } from "@/app/api/__generated__/models/credentialsMetaInput";
("use client");

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
  inputValues: Record<string, unknown>;
  setInputValue: (key: string, value: unknown) => void;
  agentInputFields: Record<string, unknown>;
  // Credentials
  inputCredentials: Record<string, CredentialsMetaInput>;
  setInputCredentialsValue: (
    key: string,
    value: CredentialsMetaInput | undefined,
  ) => void;
  agentCredentialsInputFields: Record<string, unknown>;
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
