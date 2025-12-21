import { create } from "zustand";
import { DefaultStateType } from "../components/NewControlPanel/NewBlockMenu/types";
import { GetV2BuilderSearchFilterAnyOfItem } from "@/app/api/__generated__/models/getV2BuilderSearchFilterAnyOfItem";

type BlockMenuStore = {
  searchQuery: string;
  searchId: string | undefined;
  defaultState: DefaultStateType;
  integration: string | undefined;
  filters: GetV2BuilderSearchFilterAnyOfItem[];

  setFilter: (filter: GetV2BuilderSearchFilterAnyOfItem) => void;
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

  setSearchQuery: (query) => set({ searchQuery: query }),
  setSearchId: (id) => set({ searchId: id }),
  setDefaultState: (state) => set({ defaultState: state }),
  setIntegration: (integration) => set({ integration }),
  setFilter: (filter) =>
    set((state) => ({ filters: [...state.filters, filter] })),
  removeFilter: (filter) =>
    set((state) => ({ filters: state.filters.filter((f) => f !== filter) })),
  reset: () =>
    set({
      searchQuery: "",
      searchId: undefined,
      defaultState: DefaultStateType.SUGGESTION,
      integration: undefined,
    }),
}));
