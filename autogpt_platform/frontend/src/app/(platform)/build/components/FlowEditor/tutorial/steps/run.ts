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
    title: "Run Your Workflow",
    text: `
      <div class="text-body text-foreground-emphasis">
        <p class="text-body font-normal text-foreground-emphasis m-0">Your workflow is saved and ready! Now let's <strong>run it</strong> to see it in action.</p>
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
    // TODO(tokens): no exact role: ring-blue-200 #BFDBFE; no fitting role for bg-zinc-100 #EFEFF0; #EFEFF0 is surface-hover, a hover fill, and this fill is not a hover state (QUALITY 41); no fitting role for ring-zinc-200 #DADADC; #DADADC is border-base (a border colour) and surface-active (an active-state fill), and this is a ring (QUALITY 41(b))
    text: `
      <div class="text-body text-foreground-emphasis">
        <p class="text-body font-normal text-foreground-emphasis m-0">Here's the <strong>output</strong> of your block!</p>
        
        <div class="mt-3 p-3 bg-info-surface ring-1 ring-blue-200 rounded-2xl">
          <p class="text-body leading-5 font-medium text-info-foreground m-0">Latest Output:</p>
          <p class="text-body-sm leading-inherit text-info-foreground m-0 mt-1">After each run, you can see the result of each block at the bottom of the block.</p>
        </div>
        
        <div class="mt-2 p-2 bg-zinc-100 ring-1 ring-zinc-200 rounded-xl">
          <p class="text-body-sm leading-inherit text-foreground-secondary m-0">The output shows:</p>
          <ul class="text-body-sm leading-inherit text-foreground-tertiary m-0 mt-1 pl-4">
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
      waitForElement(TUTORIAL_SELECTORS.FIRST_CALCULATOR_NODE_OUTPUT, 20000)
        .then(async () => {
          fitViewToScreen();
          // fitView animates the viewport over 800ms. Wait for it to settle so
          // Shepherd measures the output at its final position before opening
          // the overlay — otherwise the spotlight lands where the output used
          // to be and the output stays hidden under the dimmed overlay.
          await new Promise((resolve) => setTimeout(resolve, 900));
        })
        .catch(() => {}),
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
