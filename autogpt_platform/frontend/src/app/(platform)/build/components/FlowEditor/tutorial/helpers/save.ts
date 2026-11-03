import { useControlPanelStore } from "../../../../stores/controlPanelStore";

export const closeSaveControl = () => {
  useControlPanelStore.getState().setSaveControlOpen(false);
  useControlPanelStore.getState().setForceOpenSave(false);
};

export const forceSaveOpen = (force: boolean) => {
  useControlPanelStore.getState().setForceOpenSave(force);
};
