import { CopilotPage } from "../../../src/playwright/pages/copilot.page";
import {
  COPILOT_SESSIONS,
  createCopilotFixture,
  SESSION_L,
} from "../lib/copilot-fixture";
import { sendMessage } from "../lib/copilot-page";
import { installRoutes } from "../lib/routes";
import { noInteraction, settle, type Scenario } from "../lib/scenario";

/** Opens a tool call, then sends only `: heartbeat` comments, one a second. */
const TOOL_WAIT_FIXTURE = "tool-call-wait.sse";
const TOOL_WAIT_PACE_MS = 1_000;
const READY_TIMEOUT_MS = 60_000;
const SETTLE_MS = 1_000;

export const copilotToolWait: Scenario = {
  name: "copilot-tool-wait",
  description:
    'Send "go" in the 200-message chat; the reply opens one tool call and then sends no tokens, so the turn sits on a long-running tool.',
  readyWhen:
    'the newest user message and the composer are visible, then network idle + 1 s; then "go" is sent and its "Running Perf lookup…" row is on screen, + 1 s',
  windowMs: 20_000,
  async setup(ctx) {
    const fixture = createCopilotFixture({
      userID: ctx.userID,
      sessions: COPILOT_SESSIONS,
      stream: ctx.stream,
      replyFixture: TOOL_WAIT_FIXTURE,
      paceMs: TOOL_WAIT_PACE_MS,
    });
    await installRoutes(ctx.page, fixture.handlers);
    const copilot = new CopilotPage(ctx.page);
    await copilot.open(SESSION_L);
    await copilot
      .getChatInput()
      .waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    await ctx.page
      .getByText(/^Turn \d+: can you help/)
      .last()
      .waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    await settle(ctx.page);
    await sendMessage(copilot, "go");
    await ctx.page
      .getByText(/^Running Perf lookup/)
      .first()
      .waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    await ctx.page.waitForTimeout(SETTLE_MS);
    return noInteraction;
  },
};
