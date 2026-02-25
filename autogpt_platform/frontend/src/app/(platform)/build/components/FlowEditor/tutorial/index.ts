import Shepherd from "shepherd.js";
import { analytics } from "@/services/analytics";
import { TUTORIAL_CONFIG } from "./constants";
import { createTutorialSteps } from "./steps";
import { injectTutorialStyles, removeTutorialStyles } from "./styles";
import {
  handleTutorialComplete,
  handleTutorialCancel,
  prefetchTutorialBlocks,
  clearPrefetchedBlocks,
} from "./helpers";
import { useNodeStore } from "../../../stores/nodeStore";
import { useEdgeStore } from "../../../stores/edgeStore";
import { useTutorialStore } from "../../../stores/tutorialStore";

let isTutorialLoading = false;
let tutorialLoadingCallback: ((loading: boolean) => void) | null = null;

export const setTutorialLoadingCallback = (
  callback: (loading: boolean) => void,
) => {
  tutorialLoadingCallback = callback;
};

export const getTutorialLoadingState = () => isTutorialLoading;

export const startTutorial = async () => {
  isTutorialLoading = true;
  tutorialLoadingCallback?.(true);

  useNodeStore.getState().setNodes([]);
  useEdgeStore.getState().setEdges([]);
  useNodeStore.getState().setNodeCounter(0);

  try {
    await prefetchTutorialBlocks();
  } finally {
    isTutorialLoading = false;
    tutorialLoadingCallback?.(false);
  }

  const tour = new Shepherd.Tour({
    useModalOverlay: TUTORIAL_CONFIG.USE_MODAL_OVERLAY,
    defaultStepOptions: {
      cancelIcon: { enabled: true },
      scrollTo: {
        behavior: TUTORIAL_CONFIG.SCROLL_BEHAVIOR,
        block: TUTORIAL_CONFIG.SCROLL_BLOCK,
      },
      classes: "new-builder-tour",
      modalOverlayOpeningRadius: 4,
    },
  });

  injectTutorialStyles();

  const steps = createTutorialSteps(tour);
  steps.forEach((step) => tour.addStep(step));

  tour.on("complete", () => {
    handleTutorialComplete();
    removeTutorialStyles();
    clearPrefetchedBlocks();
    useTutorialStore.getState().setIsTutorialRunning(false);
  });

  tour.on("cancel", () => {
    handleTutorialCancel(tour);
    removeTutorialStyles();
    clearPrefetchedBlocks();
    useTutorialStore.getState().setIsTutorialRunning(false);
  });

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
