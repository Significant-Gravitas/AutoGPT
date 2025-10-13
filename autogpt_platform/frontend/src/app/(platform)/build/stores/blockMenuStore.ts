import { create } from "zustand";
import { DefaultStateType } from "../components/NewBlockMenu/types";

type BlockMenuStore = {
  searchQuery: string;
  searchId: string | undefined;
  defaultState: DefaultStateType;
  integration: string | undefined;

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

  setSearchQuery: (query) => set({ searchQuery: query }),
  setSearchId: (id) => set({ searchId: id }),
  setDefaultState: (state) => set({ defaultState: state }),
  setIntegration: (integration) => set({ integration }),
  reset: () =>
    set({
      searchQuery: "",
      searchId: undefined,
      defaultState: DefaultStateType.SUGGESTION,
      integration: undefined,
    }),
}));
