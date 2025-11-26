/**
 * Run steps - Steps 19-21
 * Run the agent and check output
 */

import { StepOptions } from "shepherd.js";
import { TUTORIAL_SELECTORS } from "../constants";
import {
  waitForElement,
  fitViewToScreen,
  removeAllHighlights,
  pulseElement,
} from "../helpers";
import { ICONS } from "../icons";
import { banner } from "../styles";

/**
 * Creates the run steps
 */
export const createRunSteps = (tour: any): StepOptions[] => [
  // STEP 19: Run Button
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

  // STEP 20: Wait for Execution
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

  // STEP 21: Check Output
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
];

