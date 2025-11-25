import { StepOptions } from "shepherd.js";
import { TUTORIAL_CONFIG, TUTORIAL_SELECTORS } from "./constants";
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
} from "./helpers";
import { ICONS } from "./icons";
import { banner } from "./styles";
import { useNodeStore } from "../../../stores/nodeStore";

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
        const CALCULATOR_BLOCK_ID = "b1ab9b19-67a6-406d-abf5-2dba76d00c79";

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
      on: "right", // or "left", "top", "bottom" depending on preferred position
    },
    beforeShowPromise: async () => {
      closeBlockMenu();
      await waitForNodeOnCanvas(5000);
      await new Promise((resolve) => setTimeout(resolve, 300));
      fitViewToScreen();
    },
    when: {
      show: () => {
        // Highlight the node
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
  // STEP 10: Save - Open Popover
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
  // STEP 11: Save - Fill Details
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
  // STEP 12: Run Button
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
  // STEP 13: Wait for Execution
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
  // STEP 14: Check Output
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
  // STEP 15: Canvas Controls
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
  // STEP 16: Keyboard Shortcuts
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
  // STEP 18: Next Steps
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
  // STEP 19: Congratulations
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
