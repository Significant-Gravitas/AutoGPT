import { TUTORIAL_SELECTORS } from "../constants";
import { useControlPanelStore } from "../../../../stores/controlPanelStore";

export const forceBlockMenuOpen = (force: boolean) => {
  useControlPanelStore.getState().setForceOpenBlockMenu(force);
};

export const openBlockMenu = () => {
  useControlPanelStore.getState().setBlockMenuOpen(true);
};

export const closeBlockMenu = () => {
  useControlPanelStore.getState().setBlockMenuOpen(false);
  useControlPanelStore.getState().setForceOpenBlockMenu(false);
};

export const clearBlockMenuSearch = () => {
  const input = document.querySelector(
    TUTORIAL_SELECTORS.BLOCKS_SEARCH_INPUT,
  ) as HTMLInputElement;
  if (input) {
    input.value = "";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }
};
