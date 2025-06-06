"use client";

import {
  Block,
  CredentialsProviderName,
  LibraryAgent,
  Provider,
  StoreAgent,
} from "@/lib/autogpt-server-api";
import { createContext, ReactNode, useContext, useState } from "react";
import { convertLibraryAgentIntoBlock } from "@/lib/utils";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";

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
  | "my_agents";

export interface Filters {
  categories: {
    blocks: boolean;
    integrations: boolean;
    marketplace_agents: boolean;
    my_agents: boolean;
    providers: boolean;
  };
  createdBy: string[];
}

export type CategoryCounts = Record<CategoryKey, number>;

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
  searchData: SearchItem[];
  setSearchData: React.Dispatch<React.SetStateAction<SearchItem[]>>;
  categoryCounts: CategoryCounts;
  setCategoryCounts: React.Dispatch<React.SetStateAction<CategoryCounts>>;
  addNode: (block: Block) => void;
  handleAddStoreAgent: ({
    creator_name,
    slug,
  }: {
    creator_name: string;
    slug: string;
  }) => Promise<void>;
  loadingSlug: string | null;
  setLoadingSlug: React.Dispatch<React.SetStateAction<string | null>>;
}

export const BlockMenuContext = createContext<BlockMenuContextType>(
  {} as BlockMenuContextType,
);

interface BlockMenuStateProviderProps {
  children: ReactNode;
  addNode: (block: Block) => void;
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
      providers: false,
    },
    createdBy: [],
  });
  const [searchData, setSearchData] = useState<SearchItem[]>([]);

  const [searchId, setSearchId] = useState<string | undefined>(undefined);

  const [categoryCounts, setCategoryCounts] = useState<CategoryCounts>({
    blocks: 0,
    integrations: 0,
    marketplace_agents: 0,
    my_agents: 0,
  });

  const [loadingSlug, setLoadingSlug] = useState<string | null>(null);

  const api = useBackendAPI();

  const handleAddStoreAgent = async ({
    creator_name,
    slug,
  }: {
    creator_name: string;
    slug: string;
  }) => {
    try {
      setLoadingSlug(slug);
      const details = await api.getStoreAgent(creator_name, slug);

      if (!details.active_version_id) {
        console.error(
          "Cannot add store agent to library: active version ID is missing or undefined",
        );
        return;
      }

      const libraryAgent = await api.addMarketplaceAgentToLibrary(
        details.active_version_id,
      );

      const block = convertLibraryAgentIntoBlock(libraryAgent);
      addNode(block);
    } catch (error) {
      console.error("Failed to add store agent:", error);
    } finally {
      setLoadingSlug(null);
    }
  };

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
        filters,
        setFilters,
        searchData,
        setSearchData,
        categoryCounts,
        setCategoryCounts,
        addNode,
        handleAddStoreAgent,
        loadingSlug,
        setLoadingSlug,
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
