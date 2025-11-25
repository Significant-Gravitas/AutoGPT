// index.ts (or tutorial.ts)
import Shepherd from "shepherd.js";
import "shepherd.js/dist/css/shepherd.css";
import { analytics } from "@/services/analytics";
import { TUTORIAL_CONFIG } from "./constants";
import { createTutorialSteps } from "./steps";
import { injectTutorialStyles, removeTutorialStyles } from "./styles";
import { handleTutorialComplete, handleTutorialCancel } from "./helpers";

/**
 * Starts the interactive tutorial
 */
export const startTutorial = () => {
  const tour = new Shepherd.Tour({
    useModalOverlay: TUTORIAL_CONFIG.USE_MODAL_OVERLAY,
    defaultStepOptions: {
      cancelIcon: { enabled: true },
      scrollTo: {
        behavior: TUTORIAL_CONFIG.SCROLL_BEHAVIOR,
        block: TUTORIAL_CONFIG.SCROLL_BLOCK,
      },
    },
  });

  // Inject tutorial styles
  injectTutorialStyles();

  // Add all steps to the tour
  const steps = createTutorialSteps(tour);
  steps.forEach((step) => tour.addStep(step));

  // Event handlers
  tour.on("complete", () => {
    handleTutorialComplete();
    removeTutorialStyles();
  });

  tour.on("cancel", () => {
    handleTutorialCancel(tour);
    removeTutorialStyles();
  });

  // Track tutorial steps with google analytics
  for (const step of tour.steps) {
    step.on("show", () => {
      console.debug("sendTutorialStep", step.id);
      analytics.sendGAEvent("event", "tutorial_step_shown", {
        value: step.id,
      });
    });
  }

  tour.start();
};
