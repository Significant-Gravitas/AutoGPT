import { CopilotPage } from "../../../src/playwright/pages/copilot.page";
import {
  COPILOT_SESSIONS,
  createCopilotFixture,
  SESSION_L,
} from "./copilot-fixture";
import { installRoutes } from "./routes";
import { settle, type ScenarioContext } from "./scenario";

export const REPLY_FIXTURE = "reply-20-lines.sse";
/** About 33 streamed chunks a second, the pace of a real reply. */
export const LIVE_PACE_MS = 30;
const READY_TIMEOUT_MS = 60_000;

interface OpenArgs {
  ctx: ScenarioContext;
  paceMs?: number;
}

/**
 * Serves the three fixture sessions, opens the 200-message one and waits
 * until its newest message and the composer are on screen.
 */
export async function openLongSession({
  ctx,
  paceMs = LIVE_PACE_MS,
}: OpenArgs) {
  const fixture = createCopilotFixture({
    userID: ctx.userID,
    sessions: COPILOT_SESSIONS,
    stream: ctx.stream,
    replyFixture: REPLY_FIXTURE,
    paceMs,
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
  return { fixture, copilot };
}

export async function sendMessage(
  copilot: CopilotPage,
  message: string,
): Promise<void> {
  const input = copilot.getChatInput();
  await input.click();
  await input.fill(message);
  await input.press("Enter");
}
