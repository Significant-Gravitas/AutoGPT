/**
 * Block menu steps - Steps 2-5
 * Opening menu, overview, search, and select calculator
 */

import { StepOptions } from "shepherd.js";
import { TUTORIAL_CONFIG, TUTORIAL_SELECTORS, BLOCK_IDS } from "../constants";
import {
  waitForElement,
  forceBlockMenuOpen,
  focusElement,
  highlightElement,
  removeAllHighlights,
  disableOtherBlocks,
  enableAllBlocks,
  pulseElement,
  highlightFirstBlockInSearch,
} from "../helpers";
import { ICONS } from "../icons";
import { banner } from "../styles";
import { useNodeStore } from "../../../../stores/nodeStore";

/**
 * Creates the block menu steps
 */
export const createBlockMenuSteps = (tour: any): StepOptions[] => [
  // STEP 2: Open Block Menu
  {
    id: "open-block-menu",
    title: "Open the Block Menu",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Let's start by opening the Block Menu.</p>
        ${banner(ICONS.ClickIcon, "Click this button to open the menu")}
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.BLOCKS_TRIGGER,
      on: "right",
    },
    advanceOn: {
      selector: TUTORIAL_SELECTORS.BLOCKS_TRIGGER,
      event: "click",
    },
    buttons: [],
    when: {
      show: () => {
        highlightElement(TUTORIAL_SELECTORS.BLOCKS_TRIGGER);
      },
      hide: () => {
        removeAllHighlights();
      },
    },
  },

  // STEP 3: Block Menu Overview
  {
    id: "block-menu-overview",
    title: "The Block Menu",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">This is the <strong>Block Menu</strong> — your toolbox for building agents.</p>
        <p class="text-sm font-medium leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.5rem;">Here you'll find:</p>
        <ul>
          <li><strong>Input Blocks</strong> — Entry points for data</li>
          <li><strong>Action Blocks</strong> — Processing and AI operations</li>
          <li><strong>Output Blocks</strong> — Results and responses</li>
          <li><strong>Integrations</strong> — Third-party service blocks</li>
          <li><strong>Library Agents</strong> — Your personal agents</li>
          <li><strong>Marketplace Agents</strong> — Community agents</li>
        </ul>
      </div>
    `,
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
        text: "Next",
        action: () => tour.next(),
      },
    ],
  },

  // STEP 4: Search for Calculator Block
  {
    id: "search-calculator",
    title: "Search for a Block",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Let's add a Calculator block to start.</p>
        ${banner(ICONS.Keyboard, "Type Calculator in the search bar")}
        <p class="text-xs font-normal leading-[1.125rem] text-zinc-500 m-0" style="margin-top: 0.5rem;">The search will filter blocks as you type.</p>
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.BLOCKS_SEARCH_INPUT_BOX,
      on: "bottom",
    },
    beforeShowPromise: () =>
      waitForElement(TUTORIAL_SELECTORS.BLOCKS_SEARCH_INPUT_BOX),
    when: {
      show: () => {
        forceBlockMenuOpen(true);
        setTimeout(() => {
          focusElement(TUTORIAL_SELECTORS.BLOCKS_SEARCH_INPUT_BOX);
        }, 100);

        const checkForCalculator = setInterval(() => {
          const calcBlock = document.querySelector(
            TUTORIAL_SELECTORS.BLOCK_CARD_CALCULATOR_IN_SEARCH,
          );
          if (calcBlock) {
            clearInterval(checkForCalculator);

            // Blur the search input to prevent further typing
            const searchInput = document.querySelector(
              TUTORIAL_SELECTORS.BLOCKS_SEARCH_INPUT,
            ) as HTMLInputElement;
            if (searchInput) {
              searchInput.blur();
            }

            disableOtherBlocks(
              TUTORIAL_SELECTORS.BLOCK_CARD_CALCULATOR_IN_SEARCH,
            );
            pulseElement(TUTORIAL_SELECTORS.BLOCK_CARD_CALCULATOR_IN_SEARCH);
            calcBlock.scrollIntoView({ behavior: "smooth", block: "center" });
            setTimeout(() => {
              tour.next();
            }, 300);
          }
        }, TUTORIAL_CONFIG.ELEMENT_CHECK_INTERVAL);

        (window as any).__tutorialCalcInterval = checkForCalculator;
      },
      hide: () => {
        if ((window as any).__tutorialCalcInterval) {
          clearInterval((window as any).__tutorialCalcInterval);
          delete (window as any).__tutorialCalcInterval;
        }
        enableAllBlocks();
      },
    },
    buttons: [],
  },

  // STEP 5: Select Calculator Block
  {
    id: "select-calculator",
    title: "Add the Calculator Block",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">You should see the <strong>Calculator</strong> block in the results.</p>
        ${banner(ICONS.ClickIcon, "Click on the Calculator block to add it")}
        ${banner(ICONS.Drag, "You can also drag blocks onto the canvas", "bg-zinc-100 ring-1 ring-zinc-600 text-zinc-700")}
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.BLOCK_CARD_CALCULATOR,
      on: "left",
    },
    beforeShowPromise: async () => {
      forceBlockMenuOpen(true);
      await waitForElement(TUTORIAL_SELECTORS.BLOCK_CARD_CALCULATOR, 5000);
      await new Promise((resolve) => setTimeout(resolve, 100));
    },
    when: {
      show: () => {
        // Highlight any visible calculator block or the first block
        const calcBlock = document.querySelector(
          TUTORIAL_SELECTORS.BLOCK_CARD_CALCULATOR,
        );
        if (calcBlock) {
          disableOtherBlocks(TUTORIAL_SELECTORS.BLOCK_CARD_CALCULATOR);
        } else {
          // Highlight first available block
          highlightFirstBlockInSearch();
        }

        // Calculator block_id from constants
        const CALCULATOR_BLOCK_ID = BLOCK_IDS.CALCULATOR;

        // Store initial node count to detect additions
        const initialNodeCount = useNodeStore.getState().nodes.length;

        // Subscribe to node store changes
        const unsubscribe = useNodeStore.subscribe((state) => {
          // Check if a new node was added
          if (state.nodes.length > initialNodeCount) {
            // Find if a Calculator node was added
            const calculatorNode = state.nodes.find(
              (node) => node.data?.block_id === CALCULATOR_BLOCK_ID,
            );

            if (calculatorNode) {
              // Unsubscribe to prevent multiple triggers
              unsubscribe();

              // Clean up and close block menu
              enableAllBlocks();
              forceBlockMenuOpen(false);
              tour.next();
            }
          }
        });

        // Store unsubscribe function on the step for cleanup in hide
        (tour.getCurrentStep() as any)._nodeUnsubscribe = unsubscribe;
      },
    },
  },
];

