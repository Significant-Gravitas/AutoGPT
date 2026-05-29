import { CSS_CLASSES, TUTORIAL_SELECTORS } from "../constants";

export const disableOtherBlocks = (targetBlockSelector: string) => {
  document
    .querySelectorAll(TUTORIAL_SELECTORS.BLOCK_CARD_PREFIX)
    .forEach((block) => {
      const isTarget = block.matches(targetBlockSelector);
      block.classList.toggle(CSS_CLASSES.DISABLE, !isTarget);
      block.classList.toggle(CSS_CLASSES.HIGHLIGHT, isTarget);
    });
};

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

export const highlightElement = (selector: string) => {
  const element = document.querySelector(selector);
  if (element) {
    element.classList.add(CSS_CLASSES.HIGHLIGHT);
  }
};

export const removeAllHighlights = () => {
  document.querySelectorAll(`.${CSS_CLASSES.HIGHLIGHT}`).forEach((el) => {
    el.classList.remove(CSS_CLASSES.HIGHLIGHT);
  });
  document.querySelectorAll(`.${CSS_CLASSES.PULSE}`).forEach((el) => {
    el.classList.remove(CSS_CLASSES.PULSE);
  });
};

export const pulseElement = (selector: string) => {
  const element = document.querySelector(selector);
  if (element) {
    element.classList.add(CSS_CLASSES.PULSE);
  }
};

export const highlightFirstBlockInSearch = () => {
  const firstBlock = document.querySelector(
    TUTORIAL_SELECTORS.BLOCK_CARD_PREFIX,
  );
  if (firstBlock) {
    firstBlock.classList.add(CSS_CLASSES.PULSE);
    firstBlock.scrollIntoView({ behavior: "smooth", block: "center" });
  }
};
