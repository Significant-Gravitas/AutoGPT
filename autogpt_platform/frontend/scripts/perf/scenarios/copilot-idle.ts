import { openLongSession } from "../lib/copilot-page";
import { noInteraction, type Scenario } from "../lib/scenario";

export const copilotIdle: Scenario = {
  name: "copilot-idle",
  description: "A 200-message chat sitting open with nothing happening.",
  readyWhen:
    "the newest user message and the composer are visible, then network idle + 1 s",
  windowMs: 20_000,
  async setup(ctx) {
    await openLongSession({ ctx });
    return noInteraction;
  },
};
