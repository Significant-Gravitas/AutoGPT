/**
 * Run steps - Steps 19-23
 * Run the agent and check output
 */

import { StepOptions } from "shepherd.js";
import { TUTORIAL_CONFIG, TUTORIAL_SELECTORS } from "../constants";
import {
  waitForElement,
  fitViewToScreen,
  highlightElement,
  removeAllHighlights,
  pulseElement,
} from "../helpers";
import { ICONS } from "../icons";
import { banner } from "../styles";
import { useTutorialStore } from "../../../../stores/tutorialStore";

// Helper to get input requirements HTML
const getInputRequirementsHtml = () => `
  <div id="input-requirements-box" class="mt-3 p-3 bg-amber-50 ring-1 ring-amber-200 rounded-2xl">
    <p id="input-requirements-title" class="text-sm font-medium text-amber-800 m-0 mb-2">⚠️ Required: Fill in input values</p>
    <ul id="input-requirements-list" class="text-[0.8125rem] text-amber-800 m-0 pl-4 space-y-1">
      <li id="req-inputs" class="flex items-center gap-2">
        <span class="req-icon">○</span> Enter values for all input fields
      </li>
    </ul>
  </div>
`;

// Helper to update input requirements box to success state
const updateInputReqToSuccess = () => {
  const reqBox = document.querySelector("#input-requirements-box");
  const reqTitle = document.querySelector("#input-requirements-title");
  const reqList = document.querySelector("#input-requirements-list");

  if (reqBox && reqTitle) {
    reqBox.classList.remove("bg-amber-50", "ring-amber-200");
    reqBox.classList.add("bg-green-50", "ring-green-200");
    reqTitle.classList.remove("text-amber-800");
    reqTitle.classList.add("text-green-800");
    reqTitle.innerHTML = "🎉 Hurray! All input values are filled!";
    if (reqList) {
      reqList.classList.add("hidden");
    }
  }
};

// Helper to update input requirements box back to warning state
const updateInputReqToWarning = () => {
  const reqBox = document.querySelector("#input-requirements-box");
  const reqTitle = document.querySelector("#input-requirements-title");
  const reqList = document.querySelector("#input-requirements-list");

  if (reqBox && reqTitle) {
    reqBox.classList.remove("bg-green-50", "ring-green-200");
    reqBox.classList.add("bg-amber-50", "ring-amber-200");
    reqTitle.classList.remove("text-green-800");
    reqTitle.classList.add("text-amber-800");
    reqTitle.innerHTML = "⚠️ Required: Fill in input values";
    if (reqList) {
      reqList.classList.remove("hidden");
    }
  }
};

// Helper to check if all required inputs have values (DOM-based check)
const checkInputsHaveValues = (): boolean => {
  // Check DOM for filled inputs in the inputs section
  const inputsSection = document.querySelector(
    '[data-id="run-input-inputs-section"]',
  );
  if (!inputsSection) return false;

  // Check all input/textarea elements in the inputs section
  const inputs = inputsSection.querySelectorAll(
    'input:not([type="hidden"]), textarea',
  );
  if (inputs.length === 0) return false;

  // Check if all visible inputs have values
  let allFilled = true;
  inputs.forEach((input) => {
    const el = input as HTMLInputElement | HTMLTextAreaElement;
    // Skip disabled or readonly inputs
    if (el.disabled || el.readOnly) return;
    if (!el.value || el.value.trim() === "") {
      allFilled = false;
    }
  });

  return allFilled;
};

/**
 * Creates the run steps
 */
export const createRunSteps = (tour: any): StepOptions[] => [
  // STEP 19: Run Button - click to open input dialog
  {
    id: "run-agent",
    title: "Run Your Agent",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Your agent is saved! Now let's <strong>run it</strong>.</p>
        ${banner(ICONS.ClickIcon, "Click the Run button", "action")}
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.RUN_BUTTON,
      on: "top",
    },
    beforeShowPromise: async () => {
      await waitForElement(TUTORIAL_SELECTORS.RUN_BUTTON, 3000).catch(() => {});
      await new Promise((resolve) => setTimeout(resolve, 500));
    },
    advanceOn: {
      selector: TUTORIAL_SELECTORS.RUN_BUTTON,
      event: "click",
    },
    when: {
      show: () => {
        highlightElement(TUTORIAL_SELECTORS.RUN_BUTTON);
      },
      hide: () => {
        removeAllHighlights();
      },
    },
    buttons: [],
  },

  // STEP 20: Input Dialog Overview - show the dialog
  {
    id: "input-dialog-overview",
    title: "Run Input Dialog",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">This is the <strong>Run Input Dialog</strong> — where you provide values for your agent.</p>
        
        <div class="mt-3 p-3 bg-blue-50 ring-1 ring-blue-200 rounded-2xl">
          <p class="text-sm font-medium text-blue-800 m-0 mb-1">📝 Input Dialog:</p>
          <p class="text-[0.8125rem] text-blue-800 m-0">This dialog lets you enter values for each input your agent expects before running.</p>
        </div>
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.RUN_INPUT_DIALOG_CONTENT,
      on: "left",
    },
    beforeShowPromise: async () => {
      // Force open the dialog
      useTutorialStore.getState().setForceOpenRunInputDialog(true);

      // Wait for dialog to appear
      await new Promise((resolve) => setTimeout(resolve, 300));
      await waitForElement(
        TUTORIAL_SELECTORS.RUN_INPUT_DIALOG_CONTENT,
        5000,
      ).catch(() => {});
      await new Promise((resolve) => setTimeout(resolve, 200));
    },
    when: {
      show: () => {
        // Keep dialog open
        useTutorialStore.getState().setForceOpenRunInputDialog(true);
      },
    },
    buttons: [
      {
        text: "Next",
        action: () => tour.next(),
      },
    ],
  },

  // STEP 21: Fill Input Values
  {
    id: "input-dialog-fill",
    title: "Fill Input Values",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Now let's fill in the input values for your agent.</p>
        
        ${getInputRequirementsHtml()}
        ${banner(ICONS.Keyboard, "Enter values in all the input fields", "action")}
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.RUN_INPUT_INPUTS_SECTION,
      on: "left",
    },
    beforeShowPromise: async () => {
      await waitForElement(
        TUTORIAL_SELECTORS.RUN_INPUT_INPUTS_SECTION,
        3000,
      ).catch(() => {});
    },
    when: {
      show: () => {
        let wasComplete = false;

        // Highlight the inputs section
        pulseElement(TUTORIAL_SELECTORS.RUN_INPUT_INPUTS_SECTION);

        // Poll for input values being filled
        const checkInterval = setInterval(() => {
          const allFilled = checkInputsHaveValues();

          // Update requirement icon
          const reqIcon = document.querySelector("#req-inputs .req-icon");
          if (reqIcon) reqIcon.textContent = allFilled ? "✓" : "○";

          // Update styling
          const reqEl = document.querySelector("#req-inputs");
          if (reqEl) {
            reqEl.classList.toggle("text-green-800", allFilled);
            reqEl.classList.toggle("text-amber-800", !allFilled);
          }

          // Update box to success state when complete
          if (allFilled && !wasComplete) {
            updateInputReqToSuccess();
            wasComplete = true;

            // Auto-advance after a short delay
            setTimeout(() => {
              tour.next();
            }, 500);
          } else if (!allFilled && wasComplete) {
            updateInputReqToWarning();
            wasComplete = false;
          }
        }, TUTORIAL_CONFIG.ELEMENT_CHECK_INTERVAL);

        (window as any).__tutorialInputCheckInterval = checkInterval;
      },
      hide: () => {
        if ((window as any).__tutorialInputCheckInterval) {
          clearInterval((window as any).__tutorialInputCheckInterval);
          delete (window as any).__tutorialInputCheckInterval;
        }
      },
    },
    buttons: [],
  },

  // STEP 22: Click Manual Run Button
  {
    id: "click-manual-run",
    title: "Execute the Agent",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Great! Now click the <strong>Manual Run</strong> button to execute your agent.</p>
        ${banner(ICONS.ClickIcon, "Click the Manual Run button", "action")}
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.RUN_INPUT_MANUAL_RUN_BUTTON,
      on: "top",
    },
    beforeShowPromise: async () => {
      await waitForElement(
        TUTORIAL_SELECTORS.RUN_INPUT_MANUAL_RUN_BUTTON,
        2000,
      ).catch(() => {});
    },
    when: {
      show: () => {
        highlightElement(TUTORIAL_SELECTORS.RUN_INPUT_MANUAL_RUN_BUTTON);
        pulseElement(TUTORIAL_SELECTORS.RUN_INPUT_MANUAL_RUN_BUTTON);
      },
      hide: () => {
        removeAllHighlights();
        // Reset the force open state when done with dialog
        useTutorialStore.getState().setForceOpenRunInputDialog(false);
      },
    },
    advanceOn: {
      selector: TUTORIAL_SELECTORS.RUN_INPUT_MANUAL_RUN_BUTTON,
      event: "click",
    },
    buttons: [],
  },

  // STEP 22: Wait for Execution
  {
    id: "wait-execution",
    title: "Processing...",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Your agent is running! Watch the block for status updates.</p>
        
        <div class="mt-3 p-3 bg-blue-50 ring-1 ring-blue-200 rounded-2xl">
          <p class="text-sm font-medium text-blue-800 m-0">Status progression:</p>
          <p class="text-[0.8125rem] text-blue-800 m-0 mt-1">Queued → Running → Completed</p>
        </div>
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

  // STEP 23: Check Output
  {
    id: "check-output",
    title: "View the Output",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">The block has finished! Check the <strong>output</strong> at the bottom of the block.</p>
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.5rem;">This shows the result of your calculation.</p>
        
        <div class="mt-3 p-3 bg-green-50 ring-1 ring-green-200 rounded-2xl">
          <p class="text-sm font-medium text-green-800 m-0">🎉 Success!</p>
          <p class="text-[0.8125rem] text-green-800 m-0 mt-1">Every block displays its output after execution.</p>
        </div>
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
];
