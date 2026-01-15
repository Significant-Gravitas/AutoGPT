import { StepOptions } from "shepherd.js";
import { createWelcomeSteps } from "./welcome";
import { createBlockMenuSteps } from "./block-menu";
import { createBlockBasicsSteps } from "./block-basics";
import { createConfigureCalculatorSteps } from "./configure-calculator";
import { createSecondCalculatorSteps } from "./second-calculator";
import { createConnectionSteps } from "./connections";
import { createSaveSteps } from "./save";
import { createRunSteps } from "./run";
import { createCompletionSteps } from "./completion";

export const createTutorialSteps = (tour: any): StepOptions[] => [
  ...createWelcomeSteps(tour),
  ...createBlockMenuSteps(tour),
  ...createBlockBasicsSteps(tour),
  ...createConfigureCalculatorSteps(tour),
  ...createSecondCalculatorSteps(tour),
  ...createConnectionSteps(tour),
  ...createSaveSteps(),
  ...createRunSteps(tour),
  ...createCompletionSteps(tour),
];
