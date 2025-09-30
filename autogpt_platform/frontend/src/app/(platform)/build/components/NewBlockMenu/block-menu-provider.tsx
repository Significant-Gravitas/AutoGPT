"use client";

import { createContext, ReactNode, useContext, useState } from "react";

export enum DefaultStateType {
  SUGGESTION = "suggestion",
  ALL_BLOCKS = "all_blocks",
  INPUT_BLOCKS = "input_blocks",
  ACTION_BLOCKS = "action_blocks",
  OUTPUT_BLOCKS = "output_blocks",
  INTEGRATIONS = "integrations",
  MARKETPLACE_AGENTS = "marketplace_agents",
  MY_AGENTS = "my_agents",
}

interface BlockMenuContextType {
  searchQuery: string;
  setSearchQuery: React.Dispatch<React.SetStateAction<string>>;
  searchId: string | undefined;
  setSearchId: React.Dispatch<React.SetStateAction<string | undefined>>;
  defaultState: DefaultStateType;
  setDefaultState: React.Dispatch<React.SetStateAction<DefaultStateType>>;
  integration: string | undefined;
  setIntegration: React.Dispatch<React.SetStateAction<string | undefined>>;
}

export const BlockMenuContext = createContext<BlockMenuContextType>(
  {} as BlockMenuContextType,
);

interface BlockMenuStateProviderProps {
  children: ReactNode;
}

export function BlockMenuStateProvider({
  children,
}: BlockMenuStateProviderProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchId, setSearchId] = useState<string | undefined>(undefined);
  const [defaultState, setDefaultState] = useState<DefaultStateType>(
    DefaultStateType.SUGGESTION,
  );
  const [integration, setIntegration] = useState<string | undefined>(undefined);

  return (
    <BlockMenuContext.Provider
      value={{
        searchQuery,
        setSearchQuery,
        searchId,
        setSearchId,
        defaultState,
        setDefaultState,
        integration,
        setIntegration,
      }}
    >
      {children}
    </BlockMenuContext.Provider>
  );
}

export function useBlockMenuContext(): BlockMenuContextType {
  const context = useContext(BlockMenuContext);
  if (!context) {
    throw new Error(
      "useBlockMenuContext must be used within a BlockMenuStateProvider",
    );
  }
  return context;
}
