import { StepOptions } from "shepherd.js";
import { TUTORIAL_CONFIG, TUTORIAL_SELECTORS, BLOCK_IDS } from "./constants";
import {
  waitForElement,
  waitForNodeOnCanvas,
  forceBlockMenuOpen,
  forceSaveOpen,
  handleTutorialSkip,
  focusElement,
  highlightElement,
  removeAllHighlights,
  disableOtherBlocks,
  enableAllBlocks,
  closeBlockMenu,
  fitViewToScreen,
  highlightFirstBlockInSearch,
  pulseElement,
  getFirstNode,
  nodeHasValues,
  addAgentIOBlocks,
  getNodeByBlockId,
  isConnectionMade,
  waitForNodesCount,
  getFormContainerSelector,
  getFormContainerElement,
} from "./helpers";
import { ICONS } from "./icons";
import { banner } from "./styles";
import { useNodeStore } from "../../../stores/nodeStore";
import { useEdgeStore } from "../../../stores/edgeStore";

/**
 * Creates the tutorial steps with the tour instance
 * Interactive tutorial that guides users through building their first agent
 */
export const createTutorialSteps = (tour: any): StepOptions[] => [
  // ==========================================
  // STEP 1: Welcome
  // ==========================================
  {
    id: "welcome",
    title: "Welcome to AutoGPT Builder! 👋🏻",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">This interactive tutorial will teach you how to build your first AI agent.</p>
        <p class="text-sm font-medium leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.75rem;">You'll learn how to:</p>
        <ul class="pl-2 text-sm pt-2">
          <li>- Add blocks to your workflow</li>
          <li>- Understand block inputs and outputs</li>
          <li>- Save and run your agent</li>
          <li>- and much more...</li>
        </ul>
        <p class="text-xs font-normal leading-[1.125rem] text-zinc-500 m-0" style="margin-top: 0.75rem;">Estimated time: 3-4 minutes</p>
      </div>
    `,
    buttons: [
      {
        text: "Skip Tutorial",
        action: () => handleTutorialSkip(tour),
        classes: "shepherd-button-secondary",
      },
      {
        text: "Let's Begin",
        action: () => tour.next(),
      },
    ],
  },

  // ==========================================
  // STEP 2: Open Block Menu
  // ==========================================
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
    buttons: [], // No buttons - user must click the trigger
    when: {
      show: () => {
        highlightElement(TUTORIAL_SELECTORS.BLOCKS_TRIGGER);
      },
      hide: () => {
        removeAllHighlights();
      },
    },
  },

  // ==========================================
  // STEP 3: Block Menu Overview
  // ==========================================
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

  // ==========================================
  // STEP 4: Search for Calculator Block
  // ==========================================
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

  // ==========================================
  // STEP 5: Select Calculator Block
  // ==========================================
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

  // ==========================================
  // STEP 6: Focus on New Block
  // ==========================================
  {
    id: "focus-new-block",
    title: "Your First Block!",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Excellent! This is your <strong>Calculator Block</strong>.</p>
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.5rem;">Let's explore how blocks work.</p>
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.REACT_FLOW_NODE,
      on: "right",
    },
    beforeShowPromise: async () => {
      closeBlockMenu();
      await waitForNodeOnCanvas(5000);
      await new Promise((resolve) => setTimeout(resolve, 300));
      fitViewToScreen();
    },
    when: {
      show: () => {
        const node = document.querySelector(TUTORIAL_SELECTORS.REACT_FLOW_NODE);
        if (node) {
          highlightElement(TUTORIAL_SELECTORS.REACT_FLOW_NODE);
        }
      },
      hide: () => {
        removeAllHighlights();
      },
    },
    buttons: [
      {
        text: "Show me",
        action: () => tour.next(),
      },
    ],
  },

  // ==========================================
  // STEP 7: Input Handles
  // ==========================================
  {
    id: "input-handles",
    title: "Input Handles",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">On the <strong>left side</strong> of the block are <strong>input handles</strong>.</p>
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.5rem;">These are where data flows <em>into</em> the block from other blocks.</p>
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.NODE_INPUT_HANDLE,
      on: "bottom",
    },
    classes: "new-builder-tour input-handles-step",
    beforeShowPromise: () =>
      waitForElement(TUTORIAL_SELECTORS.NODE_INPUT_HANDLE, 3000).catch(
        () => {},
      ),
    buttons: [
      {
        text: "Back",
        action: () => tour.back(),
        classes: "shepherd-button-secondary",
      },
      {
        text: "Next",
        action: () => tour.next(),
      },
    ],
  },

  // ==========================================
  // STEP 8: Output Handles
  // ==========================================
  {
    id: "output-handles",
    title: "Output Handles",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">On the <strong>right side</strong> is the <strong>output handle</strong>.</p>
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.5rem;">This is where the result flows <em>out</em> to connect to other blocks.</p>
        ${banner(ICONS.Drag, "You can drag from output to input handler to connect blocks")}
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.NODE_OUTPUT_HANDLE,
      on: "right",
    },
    beforeShowPromise: () =>
      waitForElement(TUTORIAL_SELECTORS.NODE_OUTPUT_HANDLE, 3000).catch(
        () => {},
      ),
    buttons: [
      {
        text: "Back",
        action: () => tour.back(),
        classes: "shepherd-button-secondary",
      },
      {
        text: "Next →",
        action: () => tour.next(),
      },
    ],
  },

  // ==========================================
  // STEP 9: Enter Values (Required)
  // ==========================================
  {
    id: "enter-values",
    title: "Enter Values",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Now let's configure the block with actual values.</p>
        
        <div class="mt-3 p-3 bg-amber-50 border border-amber-200 rounded-lg">
          <p class="text-sm font-medium text-amber-800 m-0 mb-2">⚠️ Required to continue:</p>
          <ul class="text-[0.8125rem] text-amber-700 m-0 pl-4 space-y-1">
            <li id="req-a" class="flex items-center gap-2">
              <span class="req-icon">○</span> Enter a number in field <strong>A</strong> (e.g., 10)
            </li>
            <li id="req-b" class="flex items-center gap-2">
              <span class="req-icon">○</span> Enter a number in field <strong>B</strong> (e.g., 5)
            </li>
            <li id="req-op" class="flex items-center gap-2">
              <span class="req-icon">○</span> Select an <strong>Operation</strong> (Add, Multiply, etc.)
            </li>
          </ul>
        </div>
        ${banner(ICONS.ClickIcon, "Fill in all the required fields above")}
      </div>
    `,
    beforeShowPromise: () => {
      fitViewToScreen();
      return Promise.resolve();
    },
    attachTo: {
      element: TUTORIAL_SELECTORS.NODE_FORM_CONTAINER,
      on: "right",
    },
    when: {
      show: () => {
        const node = getFirstNode();
        if (node) {
          highlightElement(`[data-id="custom-node-${node.id}"]`);
        }

        // Start polling to update requirements UI and button visibility
        const checkInterval = setInterval(() => {
          const node = getFirstNode();
          if (!node) return;

          const hardcodedValues = node.data?.hardcodedValues || {};
          const hasA =
            hardcodedValues.a !== undefined &&
            hardcodedValues.a !== null &&
            hardcodedValues.a !== "";
          const hasB =
            hardcodedValues.b !== undefined &&
            hardcodedValues.b !== null &&
            hardcodedValues.b !== "";
          const hasOp =
            hardcodedValues.operation !== undefined &&
            hardcodedValues.operation !== null &&
            hardcodedValues.operation !== "";

          // Update requirement icons
          const reqA = document.querySelector("#req-a .req-icon");
          const reqB = document.querySelector("#req-b .req-icon");
          const reqOp = document.querySelector("#req-op .req-icon");

          if (reqA) reqA.textContent = hasA ? "✓" : "○";
          if (reqB) reqB.textContent = hasB ? "✓" : "○";
          if (reqOp) reqOp.textContent = hasOp ? "✓" : "○";

          // Update styling for completed items
          document
            .querySelector("#req-a")
            ?.classList.toggle("text-green-700", hasA);
          document
            .querySelector("#req-b")
            ?.classList.toggle("text-green-700", hasB);
          document
            .querySelector("#req-op")
            ?.classList.toggle("text-green-700", hasOp);

          // Show/hide the next button based on completion
          const nextBtn = document.querySelector(
            ".shepherd-button-primary",
          ) as HTMLButtonElement;
          if (nextBtn) {
            const allComplete = hasA && hasB && hasOp;
            nextBtn.style.opacity = allComplete ? "1" : "0.5";
            nextBtn.style.pointerEvents = allComplete ? "auto" : "none";
          }
        }, 300);

        // Store interval ID for cleanup
        (window as any).__tutorialCheckInterval = checkInterval;
      },
      hide: () => {
        removeAllHighlights();
        // Clean up interval
        if ((window as any).__tutorialCheckInterval) {
          clearInterval((window as any).__tutorialCheckInterval);
          delete (window as any).__tutorialCheckInterval;
        }
      },
    },
    buttons: [
      {
        text: "Back",
        action: () => tour.back(),
        classes: "shepherd-button-secondary",
      },
      {
        text: "Continue",
        action: () => {
          const node = getFirstNode();
          if (!node) return;

          const hardcodedValues = node.data?.hardcodedValues || {};
          const hasA =
            hardcodedValues.a !== undefined &&
            hardcodedValues.a !== null &&
            hardcodedValues.a !== "";
          const hasB =
            hardcodedValues.b !== undefined &&
            hardcodedValues.b !== null &&
            hardcodedValues.b !== "";
          const hasOp =
            hardcodedValues.operation !== undefined &&
            hardcodedValues.operation !== null &&
            hardcodedValues.operation !== "";

          if (hasA && hasB && hasOp) {
            tour.next();
          }
        },
        classes: "shepherd-button-primary",
      },
    ],
  },

  // ==========================================
  // STEP 10: Ask Permission to Add Agent IO Blocks
  // ==========================================
  {
    id: "ask-add-agent-io-blocks",
    title: "Add Agent Input & Output",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Great job configuring the Calculator!</p>
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.5rem;">Now we need to add <strong>Agent Input</strong> and <strong>Agent Output</strong> blocks to complete your agent.</p>
        
        <div class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p class="text-sm font-medium text-blue-800 m-0 mb-1">These blocks are essential:</p>
          <ul class="text-[0.8125rem] text-blue-700 m-0 pl-4">
            <li>• <strong>Agent Input</strong> — Receives data when the agent runs</li>
            <li>• <strong>Agent Output</strong> — Returns the result to the user</li>
          </ul>
        </div>
        
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.75rem;">Can I add these blocks for you?</p>
      </div>
    `,
    buttons: [
      {
        text: "Back",
        action: () => tour.back(),
        classes: "shepherd-button-secondary",
      },
      {
        text: "Yes, Add Blocks",
        action: () => tour.next(),
      },
    ],
  },

  // ==========================================
  // STEP 11: Blocks Added Confirmation
  // ==========================================
  {
    id: "blocks-added",
    title: "Blocks Added! ✅",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">I've added <strong>Agent Input</strong> and <strong>Agent Output</strong> blocks to your canvas.</p>
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.5rem;">Now let's configure them and connect everything together.</p>
        <div class="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg">
          <p class="text-sm font-medium text-green-800 m-0">You now have 3 blocks:</p>
          <ul class="text-[0.8125rem] text-green-700 m-0 pl-4 mt-1">
            <li>• Agent Input (for receiving data)</li>
            <li>• Calculator (processes data)</li>
            <li>• Agent Output (for sending results)</li>
          </ul>
        </div>
      </div>
    `,
    beforeShowPromise: async () => {
      // Add the blocks programmatically when this step shows
      addAgentIOBlocks();
      await waitForNodesCount(3, 5000); // Wait for 3 nodes (Calculator + Input + Output)
      await new Promise((resolve) => setTimeout(resolve, 500));
      fitViewToScreen();
    },
    buttons: [
      {
        text: "Let's configure them",
        action: () => tour.next(),
      },
    ],
  },

  // ==========================================
  // STEP 12: Configure Agent Input Name
  // ==========================================
  {
    id: "configure-input-name",
    title: "Configure Agent Input",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">First, let's set up the <strong>Agent Input</strong> block.</p>
        
        <div class="mt-3 p-3 bg-amber-50 border border-amber-200 rounded-lg">
          <p class="text-sm font-medium text-amber-800 m-0 mb-2">⚠️ Required:</p>
          <ul class="text-[0.8125rem] text-amber-700 m-0 pl-4">
            <li id="req-input-name" class="flex items-center gap-2">
              <span class="req-icon">○</span> Enter a <strong>Name</strong> for the input (e.g., "number_a")
            </li>
          </ul>
        </div>
        ${banner(ICONS.ClickIcon, "Fill in the Name field in this block")}
      </div>
    `,
    // NO attachTo here - we'll set it dynamically
    beforeShowPromise: async () => {
      fitViewToScreen();
      await new Promise((resolve) => setTimeout(resolve, 300));

      const inputNode = getNodeByBlockId(BLOCK_IDS.AGENT_INPUT);
      if (inputNode) {
        const formSelector = `[data-id="form-creator-container-${inputNode.id}"]`;
        await waitForElement(formSelector, 3000).catch(() => {});
      }
      return Promise.resolve();
    },
    when: {
      show: () => {
        const inputNode = getNodeByBlockId(BLOCK_IDS.AGENT_INPUT);
        if (inputNode) {
          highlightElement(`[data-id="custom-node-${inputNode.id}"]`);

          // Get the form container and manually position the popover
          const formContainer = document.querySelector(
            `[data-id="form-creator-container-${inputNode.id}"]`,
          );

          // Get the Shepherd popover element and position it
          const popover = document.querySelector(".shepherd-element");
          if (formContainer && popover) {
            const rect = formContainer.getBoundingClientRect();
            (popover as HTMLElement).style.position = "fixed";
            (popover as HTMLElement).style.left = `${rect.left - 320}px`; // Position to the left
            (popover as HTMLElement).style.top = `${rect.top}px`;
          }
        }

        // Poll for name being set
        const checkInterval = setInterval(() => {
          const node = getNodeByBlockId(BLOCK_IDS.AGENT_INPUT);
          if (!node) return;

          const hardcodedValues = node.data?.hardcodedValues || {};
          const hasName =
            hardcodedValues.name && hardcodedValues.name.trim() !== "";

          const reqName = document.querySelector("#req-input-name .req-icon");
          if (reqName) reqName.textContent = hasName ? "✓" : "○";
          document
            .querySelector("#req-input-name")
            ?.classList.toggle("text-green-700", hasName);

          const nextBtn = document.querySelector(
            ".shepherd-button-primary",
          ) as HTMLButtonElement;
          if (nextBtn) {
            nextBtn.style.opacity = hasName ? "1" : "0.5";
            nextBtn.style.pointerEvents = hasName ? "auto" : "none";
          }
        }, 300);

        (window as any).__tutorialInputNameInterval = checkInterval;
      },
      hide: () => {
        removeAllHighlights();
        if ((window as any).__tutorialInputNameInterval) {
          clearInterval((window as any).__tutorialInputNameInterval);
          delete (window as any).__tutorialInputNameInterval;
        }
      },
    },
    buttons: [
      {
        text: "Back",
        action: () => tour.back(),
        classes: "shepherd-button-secondary",
      },
      {
        text: "Continue",
        action: () => {
          const node = getNodeByBlockId(BLOCK_IDS.AGENT_INPUT);
          if (!node) return;
          const hasName = node.data?.hardcodedValues?.name?.trim();
          if (hasName) tour.next();
        },
        classes: "shepherd-button-primary",
      },
    ],
  },

  // ==========================================
  // STEP 13: Configure Agent Output Name
  // ==========================================
  {
    id: "configure-output-name",
    title: "Configure Agent Output",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Now, let's set up the <strong>Agent Output</strong> block.</p>
        
        <div class="mt-3 p-3 bg-amber-50 border border-amber-200 rounded-lg">
          <p class="text-sm font-medium text-amber-800 m-0 mb-2">⚠️ Required:</p>
          <ul class="text-[0.8125rem] text-amber-700 m-0 pl-4">
            <li id="req-output-name" class="flex items-center gap-2">
              <span class="req-icon">○</span> Enter a <strong>Name</strong> for the output (e.g., "result")
            </li>
          </ul>
        </div>
        ${banner(ICONS.ClickIcon, "Fill in the Name field in this block")}
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.REACT_FLOW_NODE,
      on: "left",
    },
    beforeShowPromise: async () => {
      fitViewToScreen();
      await new Promise((resolve) => setTimeout(resolve, 300));

      // Get the Agent Output node and update the step's attachTo element
      const outputNode = getNodeByBlockId(BLOCK_IDS.AGENT_OUTPUT);
      if (outputNode) {
        const formSelector = `[data-id="form-creator-container-${outputNode.id}"]`;
        await waitForElement(formSelector, 3000).catch(() => {});

        // Update the step's target element before it shows
        const step = tour.getCurrentStep();
        if (step && step.options) {
          step.options.attachTo = {
            element: formSelector,
            on: "left",
          };
        }
      }
      return Promise.resolve();
    },
    when: {
      show: () => {
        const outputNode = getNodeByBlockId(BLOCK_IDS.AGENT_OUTPUT);
        if (outputNode) {
          highlightElement(`[data-id="custom-node-${outputNode.id}"]`);
        }

        // Poll for name being set
        const checkInterval = setInterval(() => {
          const node = getNodeByBlockId(BLOCK_IDS.AGENT_OUTPUT);
          if (!node) return;

          const hardcodedValues = node.data?.hardcodedValues || {};
          const hasName =
            hardcodedValues.name && hardcodedValues.name.trim() !== "";

          // Update requirement icon
          const reqName = document.querySelector("#req-output-name .req-icon");
          if (reqName) reqName.textContent = hasName ? "✓" : "○";
          document
            .querySelector("#req-output-name")
            ?.classList.toggle("text-green-700", hasName);

          // Show/hide next button
          const nextBtn = document.querySelector(
            ".shepherd-button-primary",
          ) as HTMLButtonElement;
          if (nextBtn) {
            nextBtn.style.opacity = hasName ? "1" : "0.5";
            nextBtn.style.pointerEvents = hasName ? "auto" : "none";
          }
        }, 300);

        (window as any).__tutorialOutputNameInterval = checkInterval;
      },
      hide: () => {
        removeAllHighlights();
        if ((window as any).__tutorialOutputNameInterval) {
          clearInterval((window as any).__tutorialOutputNameInterval);
          delete (window as any).__tutorialOutputNameInterval;
        }
      },
    },
    buttons: [
      {
        text: "Back",
        action: () => tour.back(),
        classes: "shepherd-button-secondary",
      },
      {
        text: "Continue",
        action: () => {
          const node = getNodeByBlockId(BLOCK_IDS.AGENT_OUTPUT);
          if (!node) return;
          const hasName = node.data?.hardcodedValues?.name?.trim();
          if (hasName) tour.next();
        },
        classes: "shepherd-button-primary",
      },
    ],
  },

  // ==========================================
  // STEP 14: Connect Agent Input to Calculator
  // ==========================================
  {
    id: "connect-input-to-calculator",
    title: "Connect Agent Input → Calculator",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Now let's connect the blocks together!</p>
        
        <div class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p class="text-sm font-medium text-blue-800 m-0 mb-2">Connect Agent Input to Calculator:</p>
          <ol class="text-[0.8125rem] text-blue-700 m-0 pl-4 space-y-1">
            <li>1. Find the <strong>output handle</strong> (right side) of Agent Input</li>
            <li>2. Drag from it to the <strong>A</strong> or <strong>B</strong> input handle (left side) of Calculator</li>
          </ol>
        </div>
        ${banner(ICONS.Drag, "Drag from Agent Input's output to Calculator's input")}
        
        <div id="connection-status-1" class="mt-3 p-2 bg-zinc-100 rounded text-center text-sm text-zinc-600">
          Waiting for connection...
        </div>
      </div>
    `,
    beforeShowPromise: async () => {
      fitViewToScreen();
      return Promise.resolve();
    },
    when: {
      show: () => {
        // Highlight both nodes
        const inputNode = getNodeByBlockId(BLOCK_IDS.AGENT_INPUT);
        const calcNode = getNodeByBlockId(BLOCK_IDS.CALCULATOR);

        if (inputNode)
          highlightElement(`[data-id="custom-node-${inputNode.id}"]`);
        if (calcNode) pulseElement(`[data-id="custom-node-${calcNode.id}"]`);

        // Subscribe to edge store changes
        const unsubscribe = useEdgeStore.subscribe(() => {
          const connected = isConnectionMade(
            BLOCK_IDS.AGENT_INPUT,
            BLOCK_IDS.CALCULATOR,
          );
          const statusEl = document.querySelector("#connection-status-1");

          if (connected && statusEl) {
            statusEl.innerHTML = "✅ Connected!";
            statusEl.classList.remove("bg-zinc-100", "text-zinc-600");
            statusEl.classList.add("bg-green-100", "text-green-700");

            // Auto-advance after brief delay
            setTimeout(() => {
              unsubscribe();
              tour.next();
            }, 500);
          }
        });

        (tour.getCurrentStep() as any)._edgeUnsubscribe = unsubscribe;
      },
      hide: () => {
        removeAllHighlights();
        const step = tour.getCurrentStep() as any;
        if (step?._edgeUnsubscribe) {
          step._edgeUnsubscribe();
        }
      },
    },
    buttons: [
      {
        text: "Back",
        action: () => tour.back(),
        classes: "shepherd-button-secondary",
      },
      {
        text: "Skip (already connected)",
        action: () => tour.next(),
        classes: "shepherd-button-secondary",
      },
    ],
  },

  // ==========================================
  // STEP 15: Connect Calculator to Agent Output
  // ==========================================
  {
    id: "connect-calculator-to-output",
    title: "Connect Calculator → Agent Output",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Almost there! Now connect the Calculator to Agent Output.</p>
        
        <div class="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p class="text-sm font-medium text-blue-800 m-0 mb-2">Connect Calculator to Agent Output:</p>
          <ol class="text-[0.8125rem] text-blue-700 m-0 pl-4 space-y-1">
            <li>1. Find the <strong>output handle</strong> (right side) of Calculator</li>
            <li>2. Drag from it to the <strong>Value</strong> input handle (left side) of Agent Output</li>
          </ol>
        </div>
        ${banner(ICONS.Drag, "Drag from Calculator's output to Agent Output's input")}
        
        <div id="connection-status-2" class="mt-3 p-2 bg-zinc-100 rounded text-center text-sm text-zinc-600">
          Waiting for connection...
        </div>
      </div>
    `,
    beforeShowPromise: async () => {
      fitViewToScreen();
      return Promise.resolve();
    },
    when: {
      show: () => {
        // Highlight both nodes
        const calcNode = getNodeByBlockId(BLOCK_IDS.CALCULATOR);
        const outputNode = getNodeByBlockId(BLOCK_IDS.AGENT_OUTPUT);

        if (calcNode)
          highlightElement(`[data-id="custom-node-${calcNode.id}"]`);
        if (outputNode)
          pulseElement(`[data-id="custom-node-${outputNode.id}"]`);

        // Subscribe to edge store changes
        const unsubscribe = useEdgeStore.subscribe(() => {
          const connected = isConnectionMade(
            BLOCK_IDS.CALCULATOR,
            BLOCK_IDS.AGENT_OUTPUT,
          );
          const statusEl = document.querySelector("#connection-status-2");

          if (connected && statusEl) {
            statusEl.innerHTML = "✅ Connected!";
            statusEl.classList.remove("bg-zinc-100", "text-zinc-600");
            statusEl.classList.add("bg-green-100", "text-green-700");

            // Auto-advance after brief delay
            setTimeout(() => {
              unsubscribe();
              tour.next();
            }, 500);
          }
        });

        (tour.getCurrentStep() as any)._edgeUnsubscribe = unsubscribe;
      },
      hide: () => {
        removeAllHighlights();
        const step = tour.getCurrentStep() as any;
        if (step?._edgeUnsubscribe) {
          step._edgeUnsubscribe();
        }
      },
    },
    buttons: [
      {
        text: "Back",
        action: () => tour.back(),
        classes: "shepherd-button-secondary",
      },
      {
        text: "Skip (already connected)",
        action: () => tour.next(),
        classes: "shepherd-button-secondary",
      },
    ],
  },

  // ==========================================
  // STEP 16: Connections Complete
  // ==========================================
  {
    id: "connections-complete",
    title: "Connections Complete! 🎉",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Excellent! Your agent workflow is now connected:</p>
        
        <div class="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg">
          <div class="flex items-center justify-center gap-2 text-sm font-medium text-green-800">
            <span>Agent Input</span>
            <span>→</span>
            <span>Calculator</span>
            <span>→</span>
            <span>Agent Output</span>
          </div>
        </div>
        
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.75rem;">Data will flow from input, through the calculator, and out as the result.</p>
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.5rem;">Now let's save your agent!</p>
      </div>
    `,
    beforeShowPromise: async () => {
      fitViewToScreen();
      return Promise.resolve();
    },
    buttons: [
      {
        text: "Save My Agent",
        action: () => tour.next(),
      },
    ],
  },

  // ==========================================
  // STEP 17: Save - Open Popover
  // ==========================================
  {
    id: "open-save",
    title: "Save Your Agent",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Before running, we need to <strong>save</strong> your agent.</p>
        ${banner(ICONS.ClickIcon, "Click the Save button")}
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.SAVE_TRIGGER,
      on: "right",
    },
    advanceOn: {
      selector: TUTORIAL_SELECTORS.SAVE_TRIGGER,
      event: "click",
    },
    beforeShowPromise: () =>
      waitForElement(TUTORIAL_SELECTORS.SAVE_TRIGGER, 3000).catch(() => {}),
    buttons: [],
    when: {
      show: () => {
        highlightElement(TUTORIAL_SELECTORS.SAVE_TRIGGER);
      },
      hide: () => {
        removeAllHighlights();
      },
    },
  },

  // ==========================================
  // STEP 18: Save - Fill Details
  // ==========================================
  {
    id: "save-details",
    title: "Name Your Agent",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Give your agent a <strong>name</strong> and optional description.</p>
        ${banner(ICONS.ClickIcon, 'Enter a name and click "Save Agent"')}
        <p class="text-xs font-normal leading-[1.125rem] text-zinc-500 m-0" style="margin-top: 0.5rem;">Example: "My Calculator Agent"</p>
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.SAVE_CONTENT,
      on: "right",
    },
    advanceOn: {
      selector: TUTORIAL_SELECTORS.SAVE_AGENT_BUTTON,
      event: "click",
    },
    beforeShowPromise: () => waitForElement(TUTORIAL_SELECTORS.SAVE_CONTENT),
    when: {
      show: () => {
        forceSaveOpen(true);
      },
      hide: () => {
        forceSaveOpen(false);
      },
    },
    buttons: [],
  },

  // ==========================================
  // STEP 19: Run Button
  // ==========================================
  {
    id: "run-agent",
    title: "Run Your Agent",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Your agent is saved! Now let's <strong>run it</strong>.</p>
        ${banner(ICONS.ClickIcon, "Click the Run button")}
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.RUN_BUTTON,
      on: "top",
    },
    advanceOn: {
      selector: TUTORIAL_SELECTORS.RUN_BUTTON,
      event: "click",
    },
    beforeShowPromise: async () => {
      await waitForElement(TUTORIAL_SELECTORS.RUN_BUTTON, 3000).catch(() => {});
      await new Promise((resolve) => setTimeout(resolve, 500));
    },
    buttons: [],
    when: {
      show: () => {
        pulseElement(TUTORIAL_SELECTORS.RUN_BUTTON);
      },
      hide: () => {
        removeAllHighlights();
      },
    },
  },

  // ==========================================
  // STEP 20: Wait for Execution
  // ==========================================
  {
    id: "wait-execution",
    title: "Processing...",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Your agent is running! Watch the block for status updates.</p>
        <p class="text-xs font-normal leading-[1.125rem] text-zinc-500 m-0" style="margin-top: 0.5rem;">The badge will show: Queued → Running → Completed</p>
      </div>
    `,
    beforeShowPromise: async () => {
      await new Promise((resolve) => setTimeout(resolve, 500));
      fitViewToScreen();
    },
    when: {
      show: () => {
        // Auto-advance when execution completes
        const checkComplete = () => {
          const completed = document.querySelector(
            TUTORIAL_SELECTORS.BADGE_COMPLETED,
          );
          const output = document.querySelector(
            TUTORIAL_SELECTORS.NODE_LATEST_OUTPUT,
          );
          if (completed || output) {
            setTimeout(() => tour.next(), 500);
          } else {
            setTimeout(checkComplete, 500);
          }
        };
        setTimeout(checkComplete, 1000);
      },
    },
    buttons: [
      {
        text: "Skip wait",
        action: () => tour.next(),
        classes: "shepherd-button-secondary",
      },
    ],
  },

  // ==========================================
  // STEP 21: Check Output
  // ==========================================
  {
    id: "check-output",
    title: "View the Output",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">The block has finished! Check the <strong>output</strong> at the bottom of the block.</p>
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.5rem;">This shows the result of your calculation.</p>
        ${banner(ICONS.ClickIcon, "Every block displays its output after execution")}
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.NODE_LATEST_OUTPUT,
      on: "top",
    },
    beforeShowPromise: () =>
      waitForElement(TUTORIAL_SELECTORS.NODE_LATEST_OUTPUT, 5000).catch(
        () => {},
      ),
    when: {
      show: () => {
        fitViewToScreen();
      },
    },
    buttons: [
      {
        text: "Next",
        action: () => tour.next(),
      },
    ],
  },

  // ==========================================
  // STEP 22: Canvas Controls
  // ==========================================
  {
    id: "canvas-controls",
    title: "Canvas Controls",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Use these controls to navigate:</p>
        <ul>
          <li><strong>+/−</strong> — Zoom in/out</li>
          <li><strong>Fit View</strong> — Center all blocks</li>
          <li><strong>Lock</strong> — Prevent accidental moves</li>
          <li><strong>Tutorial</strong> — Restart this anytime</li>
        </ul>
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.CUSTOM_CONTROLS,
      on: "right",
    },
    beforeShowPromise: () =>
      waitForElement(TUTORIAL_SELECTORS.CUSTOM_CONTROLS, 3000).catch(() => {}),
    buttons: [
      {
        text: "Next",
        action: () => tour.next(),
      },
    ],
  },

  // ==========================================
  // STEP 23: Keyboard Shortcuts
  // ==========================================
  {
    id: "keyboard-shortcuts",
    title: "Keyboard Shortcuts",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Speed up your workflow with shortcuts:</p>
        <ul>
          <li><code>Ctrl/Cmd + Z</code> — Undo</li>
          <li><code>Ctrl/Cmd + Y</code> — Redo</li>
          <li><code>Ctrl/Cmd + C</code> — Copy block</li>
          <li><code>Ctrl/Cmd + V</code> — Paste block</li>
          <li><code>Delete</code> — Remove selected</li>
        </ul>
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.UNDO_BUTTON,
      on: "right",
    },
    beforeShowPromise: () =>
      waitForElement(TUTORIAL_SELECTORS.UNDO_BUTTON, 3000).catch(() => {}),
    buttons: [
      {
        text: "Next",
        action: () => tour.next(),
      },
    ],
  },

  // ==========================================
  // STEP 24: Next Steps
  // ==========================================
  {
    id: "next-steps",
    title: "What's Next?",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">You've built and run your first agent!</p>
        <p class="text-sm font-medium leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.75rem;">To build more complex agents:</p>
        <ul>
          <li>Add multiple blocks and connect them</li>
          <li>Try <strong>AI blocks</strong> for intelligent processing</li>
          <li>Explore <strong>Integrations</strong> for external services</li>
          <li>Use <strong>Marketplace Agents</strong> as starting points</li>
        </ul>
      </div>
    `,
    buttons: [
      {
        text: "Next",
        action: () => tour.next(),
      },
    ],
  },

  // ==========================================
  // STEP 25: Congratulations
  // ==========================================
  {
    id: "congratulations",
    title: "Congratulations!",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">You've completed the AutoGPT Builder tutorial!</p>
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.75rem;">You now know how to:</p>
        <ul>
          <li>${ICONS.ClickIcon} Add blocks to the canvas</li>
          <li>${ICONS.ClickIcon} Configure block inputs and form fields</li>
          <li>${ICONS.ClickIcon} Connect blocks together</li>
          <li>${ICONS.ClickIcon} Save and run agents</li>
          <li>${ICONS.ClickIcon} View execution outputs</li>
        </ul>
        <p class="text-sm font-medium leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.75rem;">Happy building!</p>
      </div>
    `,
    when: {
      show: () => {
        const modal = document.querySelector(
          ".shepherd-modal-overlay-container",
        );
        if (modal) {
          (modal as HTMLElement).style.opacity = "0.3";
        }
      },
    },
    buttons: [
      {
        text: "Restart",
        action: () => {
          tour.cancel();
          setTimeout(() => tour.start(), 100);
        },
        classes: "shepherd-button-secondary",
      },
      {
        text: "Finish",
        action: () => tour.complete(),
      },
    ],
  },
];
