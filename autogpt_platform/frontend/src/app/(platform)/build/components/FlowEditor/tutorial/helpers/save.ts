/**
 * Save control helpers for the tutorial
 */

import { TUTORIAL_SELECTORS } from "../constants";
import { useControlPanelStore } from "../../../../stores/controlPanelStore";

/**
 * Opens the save control popover
 */
export const openSaveControl = () => {
  useControlPanelStore.getState().setSaveControlOpen(true);
};

/**
 * Closes the save control popover
 */
export const closeSaveControl = () => {
  useControlPanelStore.getState().setSaveControlOpen(false);
  useControlPanelStore.getState().setForceOpenSave(false);
};

/**
 * Forces the save control to stay open during tutorial
 */
export const forceSaveOpen = (force: boolean) => {
  useControlPanelStore.getState().setForceOpenSave(force);
};

/**
 * Simulates a click on the save button
 */
export const clickSaveButton = () => {
  const saveButton = document.querySelector(
    TUTORIAL_SELECTORS.SAVE_AGENT_BUTTON,
  ) as HTMLButtonElement;
  if (saveButton && !saveButton.disabled) {
    saveButton.click();
  }
};

/**
 * Check if the agent has been saved (by checking if version exists)
 */
export const isAgentSaved = (): boolean => {
  const versionInput = document.querySelector(
    '[data-testid="save-control-version-output"]',
  ) as HTMLInputElement;
  return !!(versionInput && versionInput.value && versionInput.value !== "-");
};

