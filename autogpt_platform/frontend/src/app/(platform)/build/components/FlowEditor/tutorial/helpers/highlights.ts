/**
 * Highlight and animation helpers for the tutorial
 */

import { CSS_CLASSES, TUTORIAL_SELECTORS } from "../constants";

/**
 * Disables all blocks except the target block
 */
export const disableOtherBlocks = (targetBlockSelector: string) => {
  document
    .querySelectorAll(TUTORIAL_SELECTORS.BLOCK_CARD_PREFIX)
    .forEach((block) => {
      const isTarget = block.matches(targetBlockSelector);
      block.classList.toggle(CSS_CLASSES.DISABLE, !isTarget);
      block.classList.toggle(CSS_CLASSES.HIGHLIGHT, isTarget);
    });
};

/**
 * Enables all blocks (removes disable and highlight classes)
 */
export const enableAllBlocks = () => {
  document
    .querySelectorAll(TUTORIAL_SELECTORS.BLOCK_CARD_PREFIX)
    .forEach((block) => {
      block.classList.remove(
        CSS_CLASSES.DISABLE,
        CSS_CLASSES.HIGHLIGHT,
        CSS_CLASSES.PULSE,
      );
    });
};

/**
 * Adds highlight class to an element
 */
export const highlightElement = (selector: string) => {
  const element = document.querySelector(selector);
  if (element) {
    element.classList.add(CSS_CLASSES.HIGHLIGHT);
  }
};

/**
 * Removes highlight from all elements
 */
export const removeAllHighlights = () => {
  document.querySelectorAll(`.${CSS_CLASSES.HIGHLIGHT}`).forEach((el) => {
    el.classList.remove(CSS_CLASSES.HIGHLIGHT);
  });
  document.querySelectorAll(`.${CSS_CLASSES.PULSE}`).forEach((el) => {
    el.classList.remove(CSS_CLASSES.PULSE);
  });
};

/**
 * Adds pulse animation to an element
 */
export const pulseElement = (selector: string) => {
  const element = document.querySelector(selector);
  if (element) {
    element.classList.add(CSS_CLASSES.PULSE);
  }
};

/**
 * Highlights the first matching block in search results
 */
export const highlightFirstBlockInSearch = () => {
  const firstBlock = document.querySelector(
    TUTORIAL_SELECTORS.BLOCK_CARD_PREFIX,
  );
  if (firstBlock) {
    firstBlock.classList.add(CSS_CLASSES.PULSE);
    // Scroll it into view
    firstBlock.scrollIntoView({ behavior: "smooth", block: "center" });
  }
};

