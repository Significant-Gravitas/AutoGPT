import { TUTORIAL_SELECTORS } from "../constants";
import { useControlPanelStore } from "../../../../stores/controlPanelStore";

export const openSaveControl = () => {
  useControlPanelStore.getState().setSaveControlOpen(true);
};

export const closeSaveControl = () => {
  useControlPanelStore.getState().setSaveControlOpen(false);
  useControlPanelStore.getState().setForceOpenSave(false);
};

export const forceSaveOpen = (force: boolean) => {
  useControlPanelStore.getState().setForceOpenSave(force);
};

export const clickSaveButton = () => {
  const saveButton = document.querySelector(
    TUTORIAL_SELECTORS.SAVE_AGENT_BUTTON,
  ) as HTMLButtonElement;
  if (saveButton && !saveButton.disabled) {
    saveButton.click();
  }
};

export const isAgentSaved = (): boolean => {
  const versionInput = document.querySelector(
    '[data-tutorial-id="save-control-version-output"]',
  ) as HTMLInputElement;
  return !!(versionInput && versionInput.value && versionInput.value !== "-");
};
