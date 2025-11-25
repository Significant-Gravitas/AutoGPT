import { StepOptions } from "shepherd.js";
import { TUTORIAL_SELECTORS } from "./constants";
import {
  waitForElement,
  forceBlockMenuOpen,
  handleTutorialSkip,
} from "./helpers";

/**
 * Creates the tutorial steps with the tour instance
 */
export const createTutorialSteps = (tour: any): StepOptions[] => [
  // Welcome step
  {
    id: "starting-step",
    title: "Welcome to the Tutorial",
    text: "This is the AutoGPT builder!",
    buttons: [
      {
        text: "Skip Tutorial",
        action: () => handleTutorialSkip(tour),
        classes: "shepherd-button-secondary",
      },
      {
        text: "Next",
        action: () => tour.next(),
      },
    ],
  },

  // Step 1: Open block menu
  {
    id: "open-block-menu",
    title: "Open the Block Menu",
    text: "Click on this button to open the block menu where you can add blocks to your flow.",
    attachTo: {
      element: TUTORIAL_SELECTORS.BLOCKS_TRIGGER,
      on: "right",
    },
    advanceOn: {
      selector: TUTORIAL_SELECTORS.BLOCKS_TRIGGER,
      event: "click",
    },
    buttons: [],
  },

  // Step 2: Block menu opened
  {
    id: "block-menu-opened",
    title: "Block Menu",
    text: "Great! This is the block menu. Here you can search for blocks, agents, integrations, and more to add to your flow.",
    attachTo: {
      element: TUTORIAL_SELECTORS.BLOCKS_CONTENT,
      on: "left",
    },
    beforeShowPromise: () => waitForElement(TUTORIAL_SELECTORS.BLOCKS_CONTENT),
    when: {
      show: () => forceBlockMenuOpen(true),
    },
    buttons: [
      {
        text: "Skip Tutorial",
        action: () => handleTutorialSkip(tour),
        classes: "shepherd-button-secondary",
      },
      {
        text: "Next",
        action: () => tour.next(),
      },
    ],
  },

  // Step 3: Search input
  {
    id: "search-input",
    title: "Search for Blocks",
    text: "Use this search bar to find specific blocks, agents, integrations, or search by keywords. Try typing to search!",
    attachTo: {
      element: TUTORIAL_SELECTORS.BLOCKS_SEARCH_INPUT,
      on: "bottom",
    },
    scrollTo: { behavior: "smooth", block: "center" },
    buttons: [
      {
        text: "Skip Tutorial",
        action: () => handleTutorialSkip(tour),
        classes: "shepherd-button-secondary",
      },
      {
        text: "Next",
        action: () => {
          forceBlockMenuOpen(false);
          tour.next();
        },
      },
    ],
  },

  // Congratulations step
  {
    id: "congratulations",
    title: "Congratulations!",
    text: "You have successfully completed the tutorial!",
    when: {
      show: () => tour.modal.hide(),
    },
    buttons: [
      {
        text: "Finish",
        action: tour.complete,
      },
    ],
  },
];
