import { StepOptions } from "shepherd.js";
import { createWelcomeSteps } from "./welcome";
import { createBlockMenuSteps } from "./block-menu";
import { createBlockBasicsSteps } from "./block-basics";
import { createConfigureCalculatorSteps } from "./configure-calculator";
import { createAgentIOSteps } from "./agent-io";
import { createConnectionSteps } from "./connections";
import { createSaveSteps } from "./save";
import { createRunSteps } from "./run";
import { createCompletionSteps } from "./completion";

export const createTutorialSteps = (tour: any): StepOptions[] => [
  ...createWelcomeSteps(tour), // Step 1
  ...createBlockMenuSteps(tour), // Steps 2-5
  ...createBlockBasicsSteps(tour), // Steps 6-8
  ...createConfigureCalculatorSteps(tour), // Step 9
  ...createAgentIOSteps(tour), // Steps 10-13
  ...createConnectionSteps(tour), // Steps 14-16
  ...createSaveSteps(tour), // Steps 17-18
  ...createRunSteps(tour), // Steps 19-21
  ...createCompletionSteps(tour), // Steps 22-25
];
