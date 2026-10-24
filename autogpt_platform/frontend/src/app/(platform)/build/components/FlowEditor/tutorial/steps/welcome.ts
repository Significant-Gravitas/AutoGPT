import { StepOptions } from "shepherd.js";
import { handleTutorialSkip } from "../helpers";

export const createWelcomeSteps = (tour: any): StepOptions[] => [
  {
    id: "welcome",
    title: "Welcome to AutoGPT Builder! 👋🏻",
    text: `
      <div class="text-body text-foreground-emphasis">
        <p class="text-body font-normal text-foreground-emphasis m-0">This interactive tutorial will teach you how to build your first AI workflow.</p>
        <p class="text-body font-medium text-foreground-emphasis m-0" style="margin-top: 0.75rem;">You'll learn how to:</p>
        <ul class="pl-2 text-body leading-5 pt-2">
          <li>- Add blocks to your workflow</li>
          <li>- Understand block inputs and outputs</li>
          <li>- Save and run your workflow</li>
          <li>- and much more...</li>
        </ul>
        <p class="text-caption font-normal text-foreground-tertiary m-0" style="margin-top: 0.75rem;">Estimated time: 3-4 minutes</p>
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
