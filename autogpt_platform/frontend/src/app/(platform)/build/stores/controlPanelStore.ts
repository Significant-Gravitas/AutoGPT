import { create } from "zustand";

type ControlPanelStore = {
  blockMenuOpen: boolean;
  saveControlOpen: boolean;
  setBlockMenuOpen: (open: boolean) => void;
  setSaveControlOpen: (open: boolean) => void;
  reset: () => void;
};

export const useControlPanelStore = create<ControlPanelStore>((set) => ({
  blockMenuOpen: false,
  saveControlOpen: false,

  setBlockMenuOpen: (open) => set({ blockMenuOpen: open }),
  setSaveControlOpen: (open) => set({ saveControlOpen: open }),
  reset: () =>
    set({
      blockMenuOpen: false,
      saveControlOpen: false,
    }),
}));
