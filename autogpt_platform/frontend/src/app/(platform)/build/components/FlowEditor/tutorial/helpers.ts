import { CSS_CLASSES, TUTORIAL_SELECTORS, TUTORIAL_CONFIG } from "./constants";
import { useControlPanelStore } from "../../../stores/controlPanelStore";
import { Key, storage } from "@/services/storage/local-storage";

/**
 * Waits for an element to appear in the DOM
 */
export const waitForElement = (selector: string): Promise<void> => {
  return new Promise((resolve) => {
    const checkElement = () => {
      if (document.querySelector(selector)) {
        resolve();
      } else {
        setTimeout(checkElement, TUTORIAL_CONFIG.ELEMENT_CHECK_INTERVAL);
      }
    };
    checkElement();
  });
};

/**
 * Triggers the fit view button to center the canvas
 */
export const fitViewToScreen = () => {
  const fitViewButton = document.querySelector(
    TUTORIAL_SELECTORS.FIT_VIEW_BUTTON,
  ) as HTMLButtonElement;
  if (fitViewButton) {
    fitViewButton.click();
  }
};

/**
 * Disables all blocks except the target block
 */
export const disableOtherBlocks = (targetBlockSelector: string) => {
  document
    .querySelectorAll(TUTORIAL_SELECTORS.BLOCK_CARD_PREFIX)
    .forEach((block) => {
      block.classList.toggle(
        CSS_CLASSES.DISABLE,
        !block.matches(targetBlockSelector),
      );
      block.classList.toggle(
        CSS_CLASSES.HIGHLIGHT,
        block.matches(targetBlockSelector),
      );
    });
};

/**
 * Enables all blocks (removes disable and highlight classes)
 */
export const enableAllBlocks = () => {
  document
    .querySelectorAll(TUTORIAL_SELECTORS.BLOCK_CARD_PREFIX)
    .forEach((block) => {
      block.classList.remove(CSS_CLASSES.DISABLE, CSS_CLASSES.HIGHLIGHT);
    });
};

/**
 * Forces the block menu to stay open during tutorial
 */
export const forceBlockMenuOpen = (force: boolean) => {
  useControlPanelStore.getState().setForceOpenBlockMenu(force);
};

/**
 * Handles tutorial cancellation
 */
export const handleTutorialCancel = (tour: any) => {
  forceBlockMenuOpen(false);
  tour.cancel();
  storage.set(Key.SHEPHERD_TOUR, "canceled");
};

/**
 * Handles tutorial skip
 */
export const handleTutorialSkip = (tour: any) => {
  forceBlockMenuOpen(false);
  tour.cancel();
  storage.set(Key.SHEPHERD_TOUR, "skipped");
};

/**
 * Handles tutorial completion
 */
export const handleTutorialComplete = () => {
  forceBlockMenuOpen(false);
  storage.set(Key.SHEPHERD_TOUR, "completed");
};
