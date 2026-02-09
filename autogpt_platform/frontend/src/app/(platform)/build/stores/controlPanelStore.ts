import { create } from "zustand";

type ControlPanelStore = {
  blockMenuOpen: boolean;
  saveControlOpen: boolean;
  forceOpenBlockMenu: boolean;
  forceOpenSave: boolean;

  setBlockMenuOpen: (open: boolean) => void;
  setSaveControlOpen: (open: boolean) => void;
  setForceOpenBlockMenu: (force: boolean) => void;
  setForceOpenSave: (force: boolean) => void;

  reset: () => void;
};

export const useControlPanelStore = create<ControlPanelStore>((set) => ({
  blockMenuOpen: false,
  saveControlOpen: false,
  forceOpenBlockMenu: false,
  forceOpenSave: false,

  setForceOpenBlockMenu: (force) => set({ forceOpenBlockMenu: force }),
  setForceOpenSave: (force) => set({ forceOpenSave: force }),
  setBlockMenuOpen: (open) => set({ blockMenuOpen: open }),
  setSaveControlOpen: (open) => set({ saveControlOpen: open }),
  reset: () =>
    set({
      blockMenuOpen: false,
      saveControlOpen: false,
      forceOpenBlockMenu: false,
      forceOpenSave: false,
    }),
}));
