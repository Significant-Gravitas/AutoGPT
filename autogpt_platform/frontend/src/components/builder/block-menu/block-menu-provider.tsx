"use client";

import {
  Block,
  CredentialsProviderName,
  LibraryAgent,
  Provider,
  StoreAgent,
} from "@/lib/autogpt-server-api";
import { createContext, ReactNode, useContext, useState } from "react";

export type SearchItem = Block | Provider | LibraryAgent | StoreAgent;

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
  integration: CredentialsProviderName | null;
  setIntegration: React.Dispatch<
    React.SetStateAction<CredentialsProviderName | null>
  >;
  searchQuery: string;
  setSearchQuery: React.Dispatch<React.SetStateAction<string>>;
  searchId: string | undefined;
  setSearchId: React.Dispatch<React.SetStateAction<string | undefined>>;
  filters: Filters;
  setFilters: React.Dispatch<React.SetStateAction<Filters>>;
  creators: string[];
  setCreators: React.Dispatch<React.SetStateAction<string[]>>;
  searchData: SearchItem[];
  setSearchData: React.Dispatch<React.SetStateAction<SearchItem[]>>;
  addNode: (
    blockId: string,
    nodeType: string,
    hardcodedValues: any | undefined,
    nodeSchema: Block | undefined,
  ) => void;
}

export const BlockMenuContext = createContext<BlockMenuContextType>(
  {} as BlockMenuContextType,
);

interface BlockMenuStateProviderProps {
  children: ReactNode;
  addNode: (
    blockId: string,
    nodeType: string,
    hardcodedValues: any | undefined,
    nodeSchema: Block | undefined,
  ) => void;
}

export function BlockMenuStateProvider({
  children,
  addNode,
}: BlockMenuStateProviderProps) {
  const [defaultState, setDefaultState] =
    useState<DefaultStateType>("suggestion");
  const [integration, setIntegration] =
    useState<CredentialsProviderName | null>(null);
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
  const [searchData, setSearchData] = useState<SearchItem[]>([]);

  const [creators, setCreators] = useState<string[]>([]);

  const [searchId, setSearchId] = useState<string | undefined>(undefined);

  return (
    <BlockMenuContext.Provider
      value={{
        defaultState,
        setDefaultState,
        integration,
        setIntegration,
        searchQuery,
        setSearchQuery,
        searchId,
        setSearchId,
        creators,
        setCreators,
        filters,
        setFilters,
        searchData,
        setSearchData,
        addNode,
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
