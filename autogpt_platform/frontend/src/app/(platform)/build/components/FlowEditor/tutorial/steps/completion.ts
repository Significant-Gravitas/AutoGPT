import { StepOptions } from "shepherd.js";

export const createCompletionSteps = (tour: any): StepOptions[] => [
  {
    id: "congratulations",
    title: "Congratulations! 🎉",
    // TODO(tokens): no exact role: ring-green-200 #94D5AC
    text: `
      <div class="text-body text-foreground-emphasis">
        <p class="text-body font-normal text-foreground-emphasis m-0">You have successfully created and run your first workflow!</p>
        
        <div class="mt-3 p-3 bg-success-surface ring-1 ring-green-200 rounded-2xl">
          <p class="text-body leading-5 font-medium text-success m-0">You learned how to:</p>
          <ul class="text-body-sm leading-inherit text-success m-0 pl-4 mt-2 space-y-1">
            <li>• Add blocks from the Block Menu</li>
            <li>• Understand input and output handles</li>
            <li>• Configure block values</li>
            <li>• Connect blocks together</li>
            <li>• Save and run your workflow</li>
            <li>• View execution status and output</li>
          </ul>
        </div>
        
        <p class="text-body font-medium text-foreground-emphasis m-0" style="margin-top: 0.75rem;">Happy building! 🚀</p>
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
        text: "Restart Tutorial",
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
