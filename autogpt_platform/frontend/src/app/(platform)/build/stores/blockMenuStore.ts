import { create } from "zustand";
import { DefaultStateType } from "../components/NewControlPanel/NewBlockMenu/types";
import { SearchResponseItemsItem } from "@/app/api/__generated__/models/searchResponseItemsItem";
import { getSearchItemType } from "../components/NewControlPanel/NewBlockMenu/BlockMenuSearchContent/helper";
import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { GetV2BuilderSearchFilterAnyOfItem } from "@/app/api/__generated__/models/getV2BuilderSearchFilterAnyOfItem";

type BlockMenuStore = {
  searchQuery: string;
  searchId: string | undefined;
  defaultState: DefaultStateType;
  integration: string | undefined;
  filters: GetV2BuilderSearchFilterAnyOfItem[];
  creators: string[];
  creators_list: string[];
  categoryCounts: Record<GetV2BuilderSearchFilterAnyOfItem, number>;

  setCategoryCounts: (
    counts: Record<GetV2BuilderSearchFilterAnyOfItem, number>,
  ) => void;
  setCreatorsList: (searchData: SearchResponseItemsItem[]) => void;
  addCreator: (creator: string) => void;
  setCreators: (creators: string[]) => void;
  removeCreator: (creator: string) => void;
  addFilter: (filter: GetV2BuilderSearchFilterAnyOfItem) => void;
  setFilters: (filters: GetV2BuilderSearchFilterAnyOfItem[]) => void;
  removeFilter: (filter: GetV2BuilderSearchFilterAnyOfItem) => void;
  setSearchQuery: (query: string) => void;
  setSearchId: (id: string | undefined) => void;
  setDefaultState: (state: DefaultStateType) => void;
  setIntegration: (integration: string | undefined) => void;
  reset: () => void;
};

export const useBlockMenuStore = create<BlockMenuStore>((set) => ({
  searchQuery: "",
  searchId: undefined,
  defaultState: DefaultStateType.SUGGESTION,
  integration: undefined,
  filters: [],
  creators: [], // creator filters that are applied to the search results
  creators_list: [], // all creators that are available to filter by
  categoryCounts: {
    blocks: 0,
    integrations: 0,
    marketplace_agents: 0,
    my_agents: 0,
  },

  setCategoryCounts: (counts) => set({ categoryCounts: counts }),
  setCreatorsList: (searchData) => {
    const marketplaceAgents = searchData.filter((item) => {
      return getSearchItemType(item).type === "store_agent";
    }) as StoreAgent[];

    const newCreators = marketplaceAgents.map((agent) => agent.creator);

    set((state) => ({
      creators_list: Array.from(
        new Set([...state.creators_list, ...newCreators]),
      ),
    }));
  },
  setCreators: (creators) => set({ creators }),
  setFilters: (filters) => set({ filters }),
  setSearchQuery: (query) => set({ searchQuery: query }),
  setSearchId: (id) => set({ searchId: id }),
  setDefaultState: (state) => set({ defaultState: state }),
  setIntegration: (integration) => set({ integration }),
  addFilter: (filter) =>
    set((state) => ({ filters: [...state.filters, filter] })),
  removeFilter: (filter) =>
    set((state) => ({ filters: state.filters.filter((f) => f !== filter) })),
  addCreator: (creator) =>
    set((state) => ({ creators: [...state.creators, creator] })),
  removeCreator: (creator) =>
    set((state) => ({ creators: state.creators.filter((c) => c !== creator) })),
  reset: () =>
    set({
      searchQuery: "",
      searchId: undefined,
      defaultState: DefaultStateType.SUGGESTION,
      integration: undefined,
    }),
}));
