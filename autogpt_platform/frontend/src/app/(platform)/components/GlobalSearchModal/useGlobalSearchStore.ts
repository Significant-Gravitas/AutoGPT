import { create } from "zustand";

interface GlobalSearchState {
  isOpen: boolean;
  openSearch: () => void;
  closeSearch: () => void;
  toggleSearch: () => void;
}

// Platform-wide open/close state for the global search palette so any page
// (and the Cmd/Ctrl+K shortcut mounted in the platform layout) can drive a
// single modal instance.
export const useGlobalSearchStore = create<GlobalSearchState>((set) => ({
  isOpen: false,
  openSearch: () => set({ isOpen: true }),
  closeSearch: () => set({ isOpen: false }),
  toggleSearch: () => set((state) => ({ isOpen: !state.isOpen })),
}));
