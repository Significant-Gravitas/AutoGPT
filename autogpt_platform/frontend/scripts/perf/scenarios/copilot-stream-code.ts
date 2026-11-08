import { SESSION_S } from "../lib/copilot-fixture";
import { openLongSession, sendMessage } from "../lib/copilot-page";
import type { Scenario } from "../lib/scenario";

const LONG_CODE_FIXTURE = "reply-long-code.sse";
const LAST_LINE = "Run it once a day.";
/** A page that highlights slowly renders the reply well after it arrived. */
const RENDERED_TIMEOUT_MS = 120_000;

export const copilotStreamCode: Scenario = {
  name: "copilot-stream-code",
  description:
    'Send "go" in the 10-message chat and stream a reply that is one 130-line code block at ~33 chunks/s.',
  readyWhen:
    "the newest user message and the composer are visible, then network idle + 1 s",
  windowMs: 1_000,
  async setup(ctx) {
    const { fixture, copilot } = await openLongSession({
      ctx,
      sessionID: SESSION_S,
      replyFixture: LONG_CODE_FIXTURE,
    });
    return {
      async run() {
        await sendMessage(copilot, "go");
        // Wait for the last line on screen, not just the stream: the window
        // must cover rendering the whole reply on both sides of a comparison.
        await ctx.page
          .getByText(LAST_LINE, { exact: true })
          .waitFor({ state: "visible", timeout: RENDERED_TIMEOUT_MS });
        await fixture.waitForTurn(1);
      },
    };
  },
};
