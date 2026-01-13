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
  ...createWelcomeSteps(tour), // Step 1
  ...createBlockMenuSteps(tour), // Steps 2-5
  ...createBlockBasicsSteps(tour), // Steps 6-8
  ...createConfigureCalculatorSteps(tour), // Step 9
  ...createSecondCalculatorSteps(tour), // Steps 10-12
  ...createConnectionSteps(tour), // Steps 13-15
  ...createSaveSteps(tour), // Steps 16-17
  ...createRunSteps(tour), // Steps 18-21
  ...createCompletionSteps(tour), // Step 22
];
