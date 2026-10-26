import { openLongSession, sendMessage } from "../lib/copilot-page";
import type { Scenario } from "../lib/scenario";

export const copilotStreamReply: Scenario = {
  name: "copilot-stream-reply",
  description:
    'Send "go" in a 200-message chat and stream the 20-line fixture reply at ~33 chunks/s.',
  readyWhen:
    "the newest user message and the composer are visible, then network idle + 1 s",
  windowMs: 1_000,
  async setup(ctx) {
    const { fixture, copilot } = await openLongSession({ ctx });
    return {
      async run() {
        await sendMessage(copilot, "go");
        await fixture.waitForTurn(1);
      },
    };
  },
};
