/**
 * Agent I/O steps - Steps 10-13
 * Add agent input/output blocks and configure them
 */

import { StepOptions } from "shepherd.js";
import { TUTORIAL_SELECTORS, BLOCK_IDS } from "../constants";
import {
  waitForElement,
  waitForNodesCount,
  fitViewToScreen,
  highlightElement,
  removeAllHighlights,
  addAgentIOBlocks,
  getNodeByBlockId,
} from "../helpers";
import { ICONS } from "../icons";
import { banner } from "../styles";

/**
 * Creates the agent I/O steps
 */
export const createAgentIOSteps = (tour: any): StepOptions[] => [
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
      addAgentIOBlocks();
      await waitForNodesCount(3, 5000);
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

  // STEP 12: Configure Agent Input Name
  {
    id: "configure-input-name",
    title: "Configure Agent Input",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">First, let's set up the <strong>Agent Input</strong> block.</p>
        
        <div class="mt-3 p-3 bg-amber-50 border border-amber-200 rounded-lg">
          <p class="text-sm font-medium text-amber-800 m-0 mb-2">⚠️ Required:</p>
          <ul class="text-[0.8125rem] text-amber-700 m-0 pl-4">
            <li id="req-input-name" class="flex items-center gap-2 text-amber-700">
              <span class="req-icon">○</span> Enter a <strong>Name</strong> for the input (e.g., "number_a")
            </li>
          </ul>
        </div>
        ${banner(ICONS.ClickIcon, "Fill in the Name field in this block")}
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.AGENT_INPUT_NODE_FORM_CONTAINER,
      on: "right",
    },
    when: {
      show: () => {
        // Get the form container and manually position the popover
        const formContainer = document.querySelector(
          TUTORIAL_SELECTORS.AGENT_INPUT_NODE_FORM_CONTAINER,
        );

        // Get the Shepherd popover element and position it
        const popover = document.querySelector(".shepherd-element");
        if (formContainer && popover) {
          const rect = formContainer.getBoundingClientRect();
          (popover as HTMLElement).style.position = "fixed";
          (popover as HTMLElement).style.left = `${rect.left - 320}px`; // Position to the left
          (popover as HTMLElement).style.top = `${rect.top}px`;
        }

        const checkInterval = setInterval(() => {
          const node = getNodeByBlockId(BLOCK_IDS.AGENT_INPUT);
          if (!node) return;

          const hardcodedValues = node.data?.hardcodedValues || {};
          const hasName =
            hardcodedValues.name && hardcodedValues.name.trim() !== "";

          const reqName = document.querySelector("#req-input-name .req-icon");
          if (reqName) reqName.textContent = hasName ? "✓" : "○";

          // Fix: Explicitly set the correct color class instead of just toggling
          const reqNameEl = document.querySelector("#req-input-name");
          if (reqNameEl) {
            if (hasName) {
              reqNameEl.classList.remove("text-amber-700");
              reqNameEl.classList.add("text-green-700");
            } else {
              reqNameEl.classList.remove("text-green-700");
              reqNameEl.classList.add("text-amber-700");
            }
          }

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

  // STEP 13: Configure Agent Output Name
  {
    id: "configure-output-name",
    title: "Configure Agent Output",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Now, let's set up the <strong>Agent Output</strong> block.</p>
        
        <div class="mt-3 p-3 bg-amber-50 border border-amber-200 rounded-lg">
          <p class="text-sm font-medium text-amber-800 m-0 mb-2">⚠️ Required:</p>
          <ul class="text-[0.8125rem] text-amber-700 m-0 pl-4">
            <li id="req-output-name" class="flex items-center gap-2 text-amber-700">
              <span class="req-icon">○</span> Enter a <strong>Name</strong> for the output (e.g., "result")
            </li>
          </ul>
        </div>
        ${banner(ICONS.ClickIcon, "Fill in the Name field in this block")}
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.AGENT_OUTPUT_NODE_FORM_CONTAINER,
      on: "top",
    },
    when: {
      show: () => {
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

          // Fix: Explicitly set the correct color class instead of just toggling
          const reqNameEl = document.querySelector("#req-output-name");
          if (reqNameEl) {
            if (hasName) {
              reqNameEl.classList.remove("text-amber-700");
              reqNameEl.classList.add("text-green-700");
            } else {
              reqNameEl.classList.remove("text-green-700");
              reqNameEl.classList.add("text-amber-700");
            }
          }

          const nextBtn = document.querySelector(
            ".shepherd-button-primary",
          ) as HTMLButtonElement;
          if (nextBtn) {
            nextBtn.style.opacity = hasName ? "1" : "0.5";
            nextBtn.style.pointerEvents = hasName ? "auto" : "none";
          }
          nextBtn.disabled = !hasName;
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
];
