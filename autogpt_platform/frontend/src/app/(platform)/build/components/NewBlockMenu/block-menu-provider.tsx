"use client";

import { createContext, ReactNode, useContext, useState } from "react";

export type DefaultStateType =
  | "suggestion"
  | "all_blocks"
  | "input_blocks"
  | "action_blocks"
  | "output_blocks"
  | "integrations"
  | "marketplace_agents"
  | "my_agents";


interface BlockMenuContextType {
  searchQuery: string;
  setSearchQuery: React.Dispatch<React.SetStateAction<string>>;
  searchId: string | undefined;
  setSearchId: React.Dispatch<React.SetStateAction<string | undefined>>;
  defaultState: DefaultStateType;
  setDefaultState: React.Dispatch<React.SetStateAction<DefaultStateType>>;
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
  const [defaultState, setDefaultState] = useState<DefaultStateType>("suggestion");

  return (
    <BlockMenuContext.Provider
      value={{
        searchQuery,
        setSearchQuery,
        searchId,
        setSearchId,
        defaultState,
        setDefaultState,
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