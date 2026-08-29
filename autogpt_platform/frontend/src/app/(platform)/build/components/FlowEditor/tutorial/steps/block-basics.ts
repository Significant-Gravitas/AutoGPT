import { StepOptions } from "shepherd.js";
import { TUTORIAL_SELECTORS } from "../constants";
import {
  waitForElement,
  waitForNodeOnCanvas,
  closeBlockMenu,
  fitViewToScreen,
  highlightElement,
  removeAllHighlights,
} from "../helpers";
import { ICONS } from "../icons";
import { banner } from "../styles";

export const createBlockBasicsSteps = (tour: any): StepOptions[] => [
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
      on: "right",
    },
    beforeShowPromise: async () => {
      closeBlockMenu();
      await waitForNodeOnCanvas(5000);
      await new Promise((resolve) => setTimeout(resolve, 300));
      fitViewToScreen();
    },
    when: {
      show: () => {
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

  {
    id: "output-handles",
    title: "Output Handles",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">On the <strong>right side</strong> is the <strong>output handle</strong>.</p>
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.5rem;">This is where the result flows <em>out</em> to connect to other blocks.</p>
        ${banner(ICONS.Drag, "You can drag from output to input handler to connect blocks", "info")}
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
];
