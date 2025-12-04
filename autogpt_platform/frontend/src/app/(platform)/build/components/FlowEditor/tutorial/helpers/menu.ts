/**
 * Block menu helpers for the tutorial
 */

import { TUTORIAL_SELECTORS } from "../constants";
import { useControlPanelStore } from "../../../../stores/controlPanelStore";

/**
 * Forces the block menu to stay open during tutorial
 */
export const forceBlockMenuOpen = (force: boolean) => {
  useControlPanelStore.getState().setForceOpenBlockMenu(force);
};

/**
 * Opens the block menu
 */
export const openBlockMenu = () => {
  useControlPanelStore.getState().setBlockMenuOpen(true);
};

/**
 * Closes the block menu
 */
export const closeBlockMenu = () => {
  useControlPanelStore.getState().setBlockMenuOpen(false);
  useControlPanelStore.getState().setForceOpenBlockMenu(false);
};

/**
 * Clears the search input in block menu
 */
export const clearBlockMenuSearch = () => {
  const input = document.querySelector(
    TUTORIAL_SELECTORS.BLOCKS_SEARCH_INPUT,
  ) as HTMLInputElement;
  if (input) {
    input.value = "";
    input.dispatchEvent(new Event("input", { bubbles: true }));
  }
};

