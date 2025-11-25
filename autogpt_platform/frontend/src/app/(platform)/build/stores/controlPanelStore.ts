import { create } from "zustand";

type ControlPanelStore = {
  blockMenuOpen: boolean;
  saveControlOpen: boolean;
  forceOpenBlockMenu: boolean;

  setBlockMenuOpen: (open: boolean) => void;
  setSaveControlOpen: (open: boolean) => void;
  setForceOpenBlockMenu: (force: boolean) => void;

  reset: () => void;
};

export const useControlPanelStore = create<ControlPanelStore>((set) => ({
  blockMenuOpen: false,
  saveControlOpen: false,
  forceOpenBlockMenu: false,

  setForceOpenBlockMenu: (force) => set({ forceOpenBlockMenu: force }),
  setBlockMenuOpen: (open) => set({ blockMenuOpen: open }),
  setSaveControlOpen: (open) => set({ saveControlOpen: open }),
  reset: () =>
    set({
      blockMenuOpen: false,
      saveControlOpen: false,
      forceOpenBlockMenu: false,
    }),
}));
