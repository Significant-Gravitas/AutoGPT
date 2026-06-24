import { StepOptions } from "shepherd.js";
import { TUTORIAL_SELECTORS } from "../constants";
import {
  waitForElement,
  fitViewToScreen,
  highlightElement,
  removeAllHighlights,
} from "../helpers";
import { ICONS } from "../icons";
import { banner } from "../styles";

export const createRunSteps = (tour: any): StepOptions[] => [
  {
    id: "press-run",
    title: "Run Your Agent",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Your agent is saved and ready! Now let's <strong>run it</strong> to see it in action.</p>
        ${banner(ICONS.ClickIcon, "Click the Run button", "action")}
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
    beforeShowPromise: () =>
      waitForElement(TUTORIAL_SELECTORS.RUN_BUTTON, 3000).catch(() => {}),
    when: {
      show: () => {
        highlightElement(TUTORIAL_SELECTORS.RUN_BUTTON);
      },
      hide: () => {
        removeAllHighlights();
        setTimeout(() => {
          fitViewToScreen();
        }, 500);
      },
    },
    buttons: [],
  },

  {
    id: "show-output",
    title: "View the Output",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Here's the <strong>output</strong> of your block!</p>
        
        <div class="mt-3 p-3 bg-blue-50 ring-1 ring-blue-200 rounded-2xl">
          <p class="text-sm font-medium text-blue-600 m-0">Latest Output:</p>
          <p class="text-[0.8125rem] text-blue-600 m-0 mt-1">After each run, you can see the result of each block at the bottom of the block.</p>
        </div>
        
        <div class="mt-2 p-2 bg-zinc-100 ring-1 ring-zinc-200 rounded-xl">
          <p class="text-[0.8125rem] text-zinc-600 m-0">The output shows:</p>
          <ul class="text-[0.8125rem] text-zinc-500 m-0 mt-1 pl-4">
            <li>• The calculated result</li>
            <li>• Execution timestamp</li>
          </ul>
        </div>
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.FIRST_CALCULATOR_NODE_OUTPUT,
      on: "top",
    },
    beforeShowPromise: () =>
      new Promise((resolve) => {
        setTimeout(() => {
          waitForElement(TUTORIAL_SELECTORS.FIRST_CALCULATOR_NODE_OUTPUT, 5000)
            .then(() => {
              fitViewToScreen();
              resolve(undefined);
            })
            .catch(resolve);
        }, 300);
      }),
    when: {
      show: () => {
        highlightElement(TUTORIAL_SELECTORS.FIRST_CALCULATOR_NODE_OUTPUT);
      },
      hide: () => {
        removeAllHighlights();
      },
    },
    buttons: [
      {
        text: "Finish Tutorial",
        action: () => tour.next(),
      },
    ],
  },
];
