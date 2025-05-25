"use client";

import { Block } from "@/lib/autogpt-server-api";
import { createContext, ReactNode, useContext, useState } from "react";

interface BaseSearchItem {
  type: "marketing_agent" | "integration_block" | "block" | "my_agent" | "ai";
}

interface MarketingAgentItem extends BaseSearchItem {
  type: "marketing_agent";
  title: string;
  image_url: string;
  creator_name: string;
  number_of_runs: number;
}

interface AIItem extends BaseSearchItem {
  type: "ai";
  title: string;
  description: string;
  ai_name: string;
}

interface BlockItem extends BaseSearchItem {
  type: "block";
  title: string;
  description: string;
}

interface IntegrationItem extends BaseSearchItem {
  type: "integration_block";
  title: string;
  description: string;
  icon_url: string;
  number_of_blocks: number;
}

interface MyAgentItem extends BaseSearchItem {
  type: "my_agent";
  title: string;
  image_url: string;
  edited_time: string;
  version: number;
}

export type SearchItem =
  | MarketingAgentItem
  | AIItem
  | BlockItem
  | IntegrationItem
  | MyAgentItem;

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
  const [searchData, setSearchData] = useState<SearchItem[]>([]);

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
