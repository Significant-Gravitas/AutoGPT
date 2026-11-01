import { useControlPanelStore } from "../../../../stores/controlPanelStore";

export const forceBlockMenuOpen = (force: boolean) => {
  useControlPanelStore.getState().setForceOpenBlockMenu(force);
};

export const closeBlockMenu = () => {
  useControlPanelStore.getState().setBlockMenuOpen(false);
  useControlPanelStore.getState().setForceOpenBlockMenu(false);
};
