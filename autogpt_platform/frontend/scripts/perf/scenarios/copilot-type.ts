import { openLongSession } from "../lib/copilot-page";
import type { Scenario } from "../lib/scenario";

const TEXT = "check the last week";
const KEY_DELAY_MS = 50;

export const copilotType: Scenario = {
  name: "copilot-type",
  description: `Type ${TEXT.length} characters into the composer of a 200-message chat, ${KEY_DELAY_MS} ms apart.`,
  readyWhen:
    "the newest user message and the composer are visible and focused, then network idle + 1 s",
  windowMs: 2_000,
  async setup(ctx) {
    const { copilot } = await openLongSession({ ctx });
    const input = copilot.getChatInput();
    await input.click();
    return {
      async run() {
        await input.pressSequentially(TEXT, { delay: KEY_DELAY_MS });
      },
    };
  },
};
