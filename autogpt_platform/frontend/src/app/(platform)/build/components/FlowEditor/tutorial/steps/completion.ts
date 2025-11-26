/**
 * Completion steps - Steps 22-25
 * Canvas controls, keyboard shortcuts, next steps, congratulations
 */

import { StepOptions } from "shepherd.js";
import { TUTORIAL_SELECTORS } from "../constants";
import { waitForElement } from "../helpers";
import { ICONS } from "../icons";

/**
 * Creates the completion steps
 */
export const createCompletionSteps = (tour: any): StepOptions[] => [
  // STEP 22: Canvas Controls
  {
    id: "canvas-controls",
    title: "Canvas Controls",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Use these controls to navigate:</p>
        <ul>
          <li><strong>+/−</strong> — Zoom in/out</li>
          <li><strong>Fit View</strong> — Center all blocks</li>
          <li><strong>Lock</strong> — Prevent accidental moves</li>
          <li><strong>Tutorial</strong> — Restart this anytime</li>
        </ul>
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.CUSTOM_CONTROLS,
      on: "right",
    },
    beforeShowPromise: () =>
      waitForElement(TUTORIAL_SELECTORS.CUSTOM_CONTROLS, 3000).catch(() => {}),
    buttons: [
      {
        text: "Next",
        action: () => tour.next(),
      },
    ],
  },

  // STEP 23: Keyboard Shortcuts
  {
    id: "keyboard-shortcuts",
    title: "Keyboard Shortcuts",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">Speed up your workflow with shortcuts:</p>
        <ul>
          <li><code>Ctrl/Cmd + Z</code> — Undo</li>
          <li><code>Ctrl/Cmd + Y</code> — Redo</li>
          <li><code>Ctrl/Cmd + C</code> — Copy block</li>
          <li><code>Ctrl/Cmd + V</code> — Paste block</li>
          <li><code>Delete</code> — Remove selected</li>
        </ul>
      </div>
    `,
    attachTo: {
      element: TUTORIAL_SELECTORS.UNDO_BUTTON,
      on: "right",
    },
    beforeShowPromise: () =>
      waitForElement(TUTORIAL_SELECTORS.UNDO_BUTTON, 3000).catch(() => {}),
    buttons: [
      {
        text: "Next",
        action: () => tour.next(),
      },
    ],
  },

  // STEP 24: Next Steps
  {
    id: "next-steps",
    title: "What's Next?",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">You've built and run your first agent!</p>
        <p class="text-sm font-medium leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.75rem;">To build more complex agents:</p>
        <ul>
          <li>Add multiple blocks and connect them</li>
          <li>Try <strong>AI blocks</strong> for intelligent processing</li>
          <li>Explore <strong>Integrations</strong> for external services</li>
          <li>Use <strong>Marketplace Agents</strong> as starting points</li>
        </ul>
      </div>
    `,
    buttons: [
      {
        text: "Next",
        action: () => tour.next(),
      },
    ],
  },

  // STEP 25: Congratulations
  {
    id: "congratulations",
    title: "Congratulations!",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">You've completed the AutoGPT Builder tutorial!</p>
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.75rem;">You now know how to:</p>
        <ul>
          <li>${ICONS.ClickIcon} Add blocks to the canvas</li>
          <li>${ICONS.ClickIcon} Configure block inputs and form fields</li>
          <li>${ICONS.ClickIcon} Connect blocks together</li>
          <li>${ICONS.ClickIcon} Save and run agents</li>
          <li>${ICONS.ClickIcon} View execution outputs</li>
        </ul>
        <p class="text-sm font-medium leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.75rem;">Happy building!</p>
      </div>
    `,
    when: {
      show: () => {
        const modal = document.querySelector(
          ".shepherd-modal-overlay-container",
        );
        if (modal) {
          (modal as HTMLElement).style.opacity = "0.3";
        }
      },
    },
    buttons: [
      {
        text: "Restart",
        action: () => {
          tour.cancel();
          setTimeout(() => tour.start(), 100);
        },
        classes: "shepherd-button-secondary",
      },
      {
        text: "Finish",
        action: () => tour.complete(),
      },
    ],
  },
];

