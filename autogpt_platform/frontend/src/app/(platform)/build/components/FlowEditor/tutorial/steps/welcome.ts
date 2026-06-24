import { StepOptions } from "shepherd.js";
import { handleTutorialSkip } from "../helpers";

export const createWelcomeSteps = (tour: any): StepOptions[] => [
  {
    id: "welcome",
    title: "Welcome to AutoGPT Builder! 👋🏻",
    text: `
      <div class="text-sm leading-[1.375rem] text-zinc-800">
        <p class="text-sm font-normal leading-[1.375rem] text-zinc-800 m-0">This interactive tutorial will teach you how to build your first AI agent.</p>
        <p class="text-sm font-medium leading-[1.375rem] text-zinc-800 m-0" style="margin-top: 0.75rem;">You'll learn how to:</p>
        <ul class="pl-2 text-sm pt-2">
          <li>- Add blocks to your workflow</li>
          <li>- Understand block inputs and outputs</li>
          <li>- Save and run your agent</li>
          <li>- and much more...</li>
        </ul>
        <p class="text-xs font-normal leading-[1.125rem] text-zinc-500 m-0" style="margin-top: 0.75rem;">Estimated time: 3-4 minutes</p>
      </div>
    `,
    buttons: [
      {
        text: "Skip Tutorial",
        action: () => handleTutorialSkip(tour),
        classes: "shepherd-button-secondary",
      },
      {
        text: "Let's Begin",
        action: () => tour.next(),
      },
    ],
  },
];
