import { StepOptions } from "shepherd.js";
import { TUTORIAL_SELECTORS } from "../constants";
import {
  waitForElement,
  highlightElement,
  removeAllHighlights,
  forceSaveOpen,
} from "../helpers";
import { ICONS } from "../icons";
import { banner } from "../styles";
export const createSaveSteps = (): StepOptions[] => [
  {
    id: "open-save",
    title: "Save Your Agent",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Before running, we need to <strong>save</strong> your agent.</p>
        ${banner(ICONS.ClickIcon, "Click the Save button", "action")}
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.SAVE_TRIGGER,
      on: "right",
    },
    advanceOn: {
      selector: TUTORIAL_SELECTORS.SAVE_TRIGGER,
      event: "click",
    },
    beforeShowPromise: () =>
      waitForElement(TUTORIAL_SELECTORS.SAVE_TRIGGER, 3000).catch(() => {}),
    buttons: [],
    when: {
      show: () => {
        highlightElement(TUTORIAL_SELECTORS.SAVE_TRIGGER);
      },
      hide: () => {
        removeAllHighlights();
      },
    },
  },

  {
    id: "save-details",
    title: "Name Your Agent",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Give your agent a <strong>name</strong> and optional description.</p>
        ${banner(ICONS.ClickIcon, 'Enter a name and click "Save Agent"', "action")}
        <p class="text-xs font-normal leading-[1.125rem] text-zinc-500 m-0" style="margin-top: 0.5rem;">Example: "My Calculator Agent"</p>
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.SAVE_CONTENT,
      on: "right",
    },
    advanceOn: {
      selector: TUTORIAL_SELECTORS.SAVE_AGENT_BUTTON,
      event: "click",
    },
    beforeShowPromise: () => waitForElement(TUTORIAL_SELECTORS.SAVE_CONTENT),
    when: {
      show: () => {
        forceSaveOpen(true);
      },
      hide: () => {
        forceSaveOpen(false);
      },
    },
    buttons: [],
  },
];
