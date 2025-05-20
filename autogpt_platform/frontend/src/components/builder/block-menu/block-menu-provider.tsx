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

export type CategoryKey =
  | "blocks"
  | "integrations"
  | "marketplace_agents"
  | "my_agents"
  | "templates";

export interface Filters {
  categories: {
    blocks: boolean;
    integrations: boolean;
    marketplace_agents: boolean;
    my_agents: boolean;
    templates: boolean;
  };
  createdBy: string[];
}

interface BlockMenuContextType {
  defaultState: DefaultStateType;
  setDefaultState: React.Dispatch<React.SetStateAction<DefaultStateType>>;
  integration: string;
  setIntegration: React.Dispatch<React.SetStateAction<string>>;
  searchQuery: string;
  setSearchQuery: React.Dispatch<React.SetStateAction<string>>;
  filters: Filters;
  setFilters: React.Dispatch<React.SetStateAction<Filters>>;
  creators: string[];
  setCreators: React.Dispatch<React.SetStateAction<string[]>>;
}

export const BlockMenuContext = createContext<BlockMenuContextType>(
  {} as BlockMenuContextType,
);

export function BlockMenuStateProvider({ children }: { children: ReactNode }) {
  const [defaultState, setDefaultState] =
    useState<DefaultStateType>("suggestion");
  const [integration, setIntegration] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [filters, setFilters] = useState<Filters>({
    categories: {
      blocks: false,
      integrations: false,
      marketplace_agents: false,
      my_agents: false,
      templates: false,
    },
    createdBy: [],
  });

  const [creators, setCreators] = useState<string[]>([]);

  return (
    <BlockMenuContext.Provider
      value={{
        defaultState,
        setDefaultState,
        integration,
        setIntegration,
        searchQuery,
        setSearchQuery,
        creators,
        setCreators,
        filters,
        setFilters,
      }}
    >
      {children}
    </BlockMenuContext.Provider>
  );
}

export function useBlockMenuContext(): BlockMenuContextType {
  const context = useContext(BlockMenuContext);
  if (!context) {
    throw new Error("Error in context of Block");
  }
  return context;
}
