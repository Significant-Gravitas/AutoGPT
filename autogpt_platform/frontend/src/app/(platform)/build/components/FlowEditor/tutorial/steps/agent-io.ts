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

// Helper to update requirement box to success state
const updateInputReqToSuccess = () => {
  const reqBox = document.querySelector("#input-requirements-box");
  const reqTitle = document.querySelector("#input-requirements-title");
  const reqList = document.querySelector("#input-requirements-list");

  if (reqBox && reqTitle) {
    reqBox.classList.remove("bg-amber-50", "ring-amber-200");
    reqBox.classList.add("bg-green-50", "ring-green-200");
    reqTitle.classList.remove("text-amber-600");
    reqTitle.classList.add("text-green-600");
    reqTitle.innerHTML = "🎉 Hurray! Input name is set!";
    if (reqList) {
      reqList.classList.add("hidden");
    }
  }
};

// Helper to update requirement box back to warning state
const updateInputReqToWarning = () => {
  const reqBox = document.querySelector("#input-requirements-box");
  const reqTitle = document.querySelector("#input-requirements-title");
  const reqList = document.querySelector("#input-requirements-list");

  if (reqBox && reqTitle) {
    reqBox.classList.remove("bg-green-50", "ring-green-200");
    reqBox.classList.add("bg-amber-50", "ring-amber-200");
    reqTitle.classList.remove("text-green-600");
    reqTitle.classList.add("text-amber-600");
    reqTitle.innerHTML = "⚠️ Required:";
    if (reqList) {
      reqList.classList.remove("hidden");
    }
  }
};

// Helper to update output requirement box to success state
const updateOutputReqToSuccess = () => {
  const reqBox = document.querySelector("#output-requirements-box");
  const reqTitle = document.querySelector("#output-requirements-title");
  const reqList = document.querySelector("#output-requirements-list");

  if (reqBox && reqTitle) {
    reqBox.classList.remove("bg-amber-50", "ring-amber-200");
    reqBox.classList.add("bg-green-50", "ring-green-200");
    reqTitle.classList.remove("text-amber-600");
    reqTitle.classList.add("text-green-600");
    reqTitle.innerHTML = "🎉 Hurray! Output name is set!";
    if (reqList) {
      reqList.classList.add("hidden");
    }
  }
};

// Helper to update output requirement box back to warning state
const updateOutputReqToWarning = () => {
  const reqBox = document.querySelector("#output-requirements-box");
  const reqTitle = document.querySelector("#output-requirements-title");
  const reqList = document.querySelector("#output-requirements-list");

  if (reqBox && reqTitle) {
    reqBox.classList.remove("bg-green-50", "ring-green-200");
    reqBox.classList.add("bg-amber-50", "ring-amber-200");
    reqTitle.classList.remove("text-green-600");
    reqTitle.classList.add("text-amber-600");
    reqTitle.innerHTML = "⚠️ Required:";
    if (reqList) {
      reqList.classList.remove("hidden");
    }
  }
};

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
        
        <div class="mt-3 p-3 bg-blue-50 ring-1 ring-blue-200 rounded-2xl">
          <p class="text-sm font-medium text-blue-600 m-0 mb-1">These blocks are essential:</p>
          <ul class="text-[0.8125rem] text-blue-600 m-0 pl-4">
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
        <div class="mt-3 p-3 bg-green-50 ring-1 ring-green-200 rounded-2xl">
          <p class="text-sm font-medium text-green-600 m-0">You now have 3 blocks:</p>
          <ul class="text-[0.8125rem] text-green-600 m-0 pl-4 mt-1">
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
        
        <div id="input-requirements-box" class="mt-3 p-3 bg-amber-50 ring-1 ring-amber-200 rounded-2xl">
          <p id="input-requirements-title" class="text-sm font-medium text-amber-600 m-0 mb-2">⚠️ Required:</p>
          <ul id="input-requirements-list" class="text-[0.8125rem] text-amber-600 m-0 pl-4">
            <li id="req-input-name" class="flex items-center gap-2 text-amber-600">
              <span class="req-icon">○</span> Enter a <strong>Name</strong> for the input (e.g., "number_a")
            </li>
          </ul>
        </div>
        ${banner(ICONS.ClickIcon, "Fill in the Name field in this block", "action")}
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

        let wasComplete = false;

        const checkInterval = setInterval(() => {
          const node = getNodeByBlockId(BLOCK_IDS.AGENT_INPUT);
          if (!node) return;

          const hardcodedValues = node.data?.hardcodedValues || {};
          const hasName =
            hardcodedValues.name && hardcodedValues.name.trim() !== "";

          const reqName = document.querySelector("#req-input-name .req-icon");
          if (reqName) reqName.textContent = hasName ? "✓" : "○";

          // Update styling for completed item
          const reqNameEl = document.querySelector("#req-input-name");
          if (reqNameEl) {
            reqNameEl.classList.toggle("text-green-600", hasName);
            reqNameEl.classList.toggle("text-amber-600", !hasName);
          }

          // Update box to success state when complete
          if (hasName && !wasComplete) {
            updateInputReqToSuccess();
            wasComplete = true;
          } else if (!hasName && wasComplete) {
            updateInputReqToWarning();
            wasComplete = false;
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
        
        <div id="output-requirements-box" class="mt-3 p-3 bg-amber-50 ring-1 ring-amber-200 rounded-2xl">
          <p id="output-requirements-title" class="text-sm font-medium text-amber-600 m-0 mb-2">⚠️ Required:</p>
          <ul id="output-requirements-list" class="text-[0.8125rem] text-amber-600 m-0 pl-4">
            <li id="req-output-name" class="flex items-center gap-2 text-amber-600">
              <span class="req-icon">○</span> Enter a <strong>Name</strong> for the output (e.g., "result")
            </li>
          </ul>
        </div>
        ${banner(ICONS.ClickIcon, "Fill in the Name field in this block", "action")}
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.NAME_FIELD_OUTPUT_NODE,
      on: "bottom",
    },
    modalOverlayOpeningPadding: 10,
    when: {
      show: () => {
        let wasComplete = false;

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

          // Update styling for completed item
          const reqNameEl = document.querySelector("#req-output-name");
          if (reqNameEl) {
            reqNameEl.classList.toggle("text-green-600", hasName);
            reqNameEl.classList.toggle("text-amber-600", !hasName);
          }

          // Update box to success state when complete
          if (hasName && !wasComplete) {
            updateOutputReqToSuccess();
            wasComplete = true;
          } else if (!hasName && wasComplete) {
            updateOutputReqToWarning();
            wasComplete = false;
          }

          const nextBtn = document.querySelector(
            ".shepherd-button-primary",
          ) as HTMLButtonElement;
          if (nextBtn) {
            nextBtn.style.opacity = hasName ? "1" : "0.5";
            nextBtn.style.pointerEvents = hasName ? "auto" : "none";
            nextBtn.disabled = !hasName;
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
];
