/**
 * Configure calculator step - Step 9
 * Enter values in calculator block
 */

import { StepOptions } from "shepherd.js";
import { TUTORIAL_SELECTORS } from "../constants";
import {
  fitViewToScreen,
  highlightElement,
  removeAllHighlights,
  getFirstNode,
} from "../helpers";
import { ICONS } from "../icons";
import { banner } from "../styles";

/**
 * Creates the configure calculator step
 */
export const createConfigureCalculatorSteps = (tour: any): StepOptions[] => [
  // STEP 9: Enter Values (Required)
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
      element: TUTORIAL_SELECTORS.CALCULATOR_NODE_FORM_CONTAINER,
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
];
