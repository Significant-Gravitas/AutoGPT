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

const getRequirementsHtml = () => `
  <div id="requirements-box" class="mt-3 p-3 bg-amber-50 ring-1 ring-amber-200 rounded-2xl">
    <p id="requirements-title" class="text-sm font-medium text-amber-600 m-0 mb-2">⚠️ Required to continue:</p>
    <ul id="requirements-list" class="text-[0.8125rem] text-amber-600 m-0 pl-4 space-y-1">
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
`;

const updateToSuccessState = () => {
  const reqBox = document.querySelector("#requirements-box");
  const reqTitle = document.querySelector("#requirements-title");
  const reqList = document.querySelector("#requirements-list");

  if (reqBox && reqTitle) {
    reqBox.classList.remove("bg-amber-50", "ring-amber-200");
    reqBox.classList.add("bg-green-50", "ring-green-200");
    reqTitle.classList.remove("text-amber-600");
    reqTitle.classList.add("text-green-600");
    reqTitle.innerHTML = "🎉 Hurray! All values are completed!";
    if (reqList) {
      reqList.classList.add("hidden");
    }
  }
};

const updateToWarningState = () => {
  const reqBox = document.querySelector("#requirements-box");
  const reqTitle = document.querySelector("#requirements-title");
  const reqList = document.querySelector("#requirements-list");

  if (reqBox && reqTitle) {
    reqBox.classList.remove("bg-green-50", "ring-green-200");
    reqBox.classList.add("bg-amber-50", "ring-amber-200");
    reqTitle.classList.remove("text-green-600");
    reqTitle.classList.add("text-amber-600");
    reqTitle.innerHTML = "⚠️ Required to continue:";
    if (reqList) {
      reqList.classList.remove("hidden");
    }
  }
};

export const createConfigureCalculatorSteps = (tour: any): StepOptions[] => [
  {
    id: "enter-values",
    title: "Enter Values",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Now let's configure the block with actual values.</p>
        ${getRequirementsHtml()}
        ${banner(ICONS.ClickIcon, "Fill in all the required fields above", "action")}
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

        let wasComplete = false;

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

          const allComplete = hasA && hasB && hasOp;

          const reqA = document.querySelector("#req-a .req-icon");
          const reqB = document.querySelector("#req-b .req-icon");
          const reqOp = document.querySelector("#req-op .req-icon");

          if (reqA) reqA.textContent = hasA ? "✓" : "○";
          if (reqB) reqB.textContent = hasB ? "✓" : "○";
          if (reqOp) reqOp.textContent = hasOp ? "✓" : "○";

          const reqAEl = document.querySelector("#req-a");
          const reqBEl = document.querySelector("#req-b");
          const reqOpEl = document.querySelector("#req-op");

          if (reqAEl) {
            reqAEl.classList.toggle("text-green-600", hasA);
            reqAEl.classList.toggle("text-amber-600", !hasA);
          }
          if (reqBEl) {
            reqBEl.classList.toggle("text-green-600", hasB);
            reqBEl.classList.toggle("text-amber-600", !hasB);
          }
          if (reqOpEl) {
            reqOpEl.classList.toggle("text-green-600", hasOp);
            reqOpEl.classList.toggle("text-amber-600", !hasOp);
          }

          if (allComplete && !wasComplete) {
            updateToSuccessState();
            wasComplete = true;
          } else if (!allComplete && wasComplete) {
            updateToWarningState();
            wasComplete = false;
          }

          const nextBtn = document.querySelector(
            ".shepherd-button-primary",
          ) as HTMLButtonElement;
          if (nextBtn) {
            nextBtn.style.opacity = allComplete ? "1" : "0.5";
            nextBtn.style.pointerEvents = allComplete ? "auto" : "none";
          }
        }, 300);

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
