import { StepOptions } from "shepherd.js";
import {
  waitForNodesCount,
  fitViewToScreen,
  highlightElement,
  removeAllHighlights,
  addSecondCalculatorBlock,
  getSecondCalculatorNode,
} from "../helpers";
import { ICONS } from "../icons";
import { banner } from "../styles";

const getSecondCalculatorFormSelector = (): string | HTMLElement => {
  const secondNode = getSecondCalculatorNode();
  if (secondNode) {
    const selector = `[data-id="form-creator-container-${secondNode.id}-node"]`;
    const element = document.querySelector(selector);
    if (element) {
      return element as HTMLElement;
    }
    return selector;
  }
  const formContainers = document.querySelectorAll(
    '[data-id^="form-creator-container-"]',
  );
  if (formContainers.length >= 2) {
    return formContainers[1] as HTMLElement;
  }
  return '[data-id^="form-creator-container-"]';
};

const getSecondCalcRequirementsHtml = () => `
  <div id="second-calc-requirements-box" class="mt-3 p-3 bg-amber-50 ring-1 ring-amber-200 rounded-2xl">
    <p id="second-calc-requirements-title" class="text-sm font-medium text-amber-600 m-0 mb-2">⚠️ Required to continue:</p>
    <ul id="second-calc-requirements-list" class="text-[0.8125rem] text-amber-600 m-0 pl-4 space-y-1">
      <li id="req2-b" class="flex items-center gap-2">
        <span class="req-icon">○</span> Enter a number in field <strong>B</strong> (e.g., 2)
      </li>
      <li id="req2-op" class="flex items-center gap-2">
        <span class="req-icon">○</span> Select an <strong>Operation</strong> (e.g., Multiply)
      </li>
    </ul>
    <p class="text-[0.75rem] text-amber-500 m-0 mt-2 italic">Note: Field A will be connected from the first Calculator's output</p>
  </div>
`;

const updateSecondCalcToSuccessState = () => {
  const reqBox = document.querySelector("#second-calc-requirements-box");
  const reqTitle = document.querySelector("#second-calc-requirements-title");
  const reqList = document.querySelector("#second-calc-requirements-list");

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

const updateSecondCalcToWarningState = () => {
  const reqBox = document.querySelector("#second-calc-requirements-box");
  const reqTitle = document.querySelector("#second-calc-requirements-title");
  const reqList = document.querySelector("#second-calc-requirements-list");

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

export const createSecondCalculatorSteps = (tour: any): StepOptions[] => [
  {
    id: "adding-second-calculator",
    title: "Adding Second Calculator",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Great job configuring the first Calculator!</p>
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.5rem;">Now let's add a <strong>second Calculator block</strong> and connect them together.</p>
        
        <div class="mt-3 p-3 bg-blue-50 ring-1 ring-blue-200 rounded-2xl">
          <p class="text-sm font-medium text-blue-600 m-0 mb-1">We'll create a chain:</p>
          <p class="text-[0.8125rem] text-blue-600 m-0">Calculator 1 → Calculator 2</p>
          <p class="text-[0.75rem] text-blue-500 m-0 mt-1 italic">The output of the first will feed into the second!</p>
        </div>
      </div>
    `,
    buttons: [
      {
        text: "Back",
        action: () => tour.back(),
        classes: "shepherd-button-secondary",
      },
      {
        text: "Add Second Calculator",
        action: () => tour.next(),
      },
    ],
  },

  {
    id: "second-calculator-added",
    title: "Second Calculator Added! ✅",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">I've added a <strong>second Calculator block</strong> to your canvas.</p>
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.5rem;">Now let's configure it and connect them together.</p>
        
        <div class="mt-3 p-3 bg-green-50 ring-1 ring-green-200 rounded-2xl">
          <p class="text-sm font-medium text-green-600 m-0">You now have 2 Calculator blocks!</p>
        </div>
      </div>
    `,
    beforeShowPromise: async () => {
      addSecondCalculatorBlock();
      await waitForNodesCount(2, 5000);
      await new Promise((resolve) => setTimeout(resolve, 500));
      fitViewToScreen();
    },
    buttons: [
      {
        text: "Let's configure it",
        action: () => tour.next(),
      },
    ],
  },

  {
    id: "configure-second-calculator",
    title: "Configure Second Calculator",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Now configure the <strong>second Calculator block</strong>.</p>
        ${getSecondCalcRequirementsHtml()}
        ${banner(ICONS.ClickIcon, "Fill in field B and select an Operation", "action")}
      </div>
    `,
    beforeShowPromise: async () => {
      fitViewToScreen();
      await new Promise<void>((resolve) => {
        const checkNode = () => {
          const secondNode = getSecondCalculatorNode();
          if (secondNode) {
            const formContainer = document.querySelector(
              `[data-id="form-creator-container-${secondNode.id}-node"]`,
            );
            if (formContainer) {
              resolve();
            } else {
              setTimeout(checkNode, 100);
            }
          } else {
            setTimeout(checkNode, 100);
          }
        };
        checkNode();
      });
    },
    attachTo: {
      element: getSecondCalculatorFormSelector,
      on: "right",
    },
    when: {
      show: () => {
        const secondNode = getSecondCalculatorNode();
        if (secondNode) {
          highlightElement(`[data-id="custom-node-${secondNode.id}"]`);
        }

        let wasComplete = false;

        const checkInterval = setInterval(() => {
          const secondNode = getSecondCalculatorNode();
          if (!secondNode) return;

          const hardcodedValues = secondNode.data?.hardcodedValues || {};
          const hasB =
            hardcodedValues.b !== undefined &&
            hardcodedValues.b !== null &&
            hardcodedValues.b !== "";
          const hasOp =
            hardcodedValues.operation !== undefined &&
            hardcodedValues.operation !== null &&
            hardcodedValues.operation !== "";

          const allComplete = hasB && hasOp;

          const reqB = document.querySelector("#req2-b .req-icon");
          const reqOp = document.querySelector("#req2-op .req-icon");

          if (reqB) reqB.textContent = hasB ? "✓" : "○";
          if (reqOp) reqOp.textContent = hasOp ? "✓" : "○";

          const reqBEl = document.querySelector("#req2-b");
          const reqOpEl = document.querySelector("#req2-op");

          if (reqBEl) {
            reqBEl.classList.toggle("text-green-600", hasB);
            reqBEl.classList.toggle("text-amber-600", !hasB);
          }
          if (reqOpEl) {
            reqOpEl.classList.toggle("text-green-600", hasOp);
            reqOpEl.classList.toggle("text-amber-600", !hasOp);
          }

          if (allComplete && !wasComplete) {
            updateSecondCalcToSuccessState();
            wasComplete = true;
          } else if (!allComplete && wasComplete) {
            updateSecondCalcToWarningState();
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

        (window as any).__tutorialSecondCalcInterval = checkInterval;
      },
      hide: () => {
        removeAllHighlights();
        if ((window as any).__tutorialSecondCalcInterval) {
          clearInterval((window as any).__tutorialSecondCalcInterval);
          delete (window as any).__tutorialSecondCalcInterval;
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
          const secondNode = getSecondCalculatorNode();
          if (!secondNode) return;

          const hardcodedValues = secondNode.data?.hardcodedValues || {};
          const hasB =
            hardcodedValues.b !== undefined &&
            hardcodedValues.b !== null &&
            hardcodedValues.b !== "";
          const hasOp =
            hardcodedValues.operation !== undefined &&
            hardcodedValues.operation !== null &&
            hardcodedValues.operation !== "";

          if (hasB && hasOp) {
            tour.next();
          }
        },
        classes: "shepherd-button-primary",
      },
    ],
  },
];
