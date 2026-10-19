import { openLongSession, sendMessage } from "../lib/copilot-page";
import { noInteraction, type Scenario } from "../lib/scenario";

/** Slow enough that the reply is still streaming for the whole window. */
const BACKGROUND_PACE_MS = 200;
const READY_TIMEOUT_MS = 30_000;
const SETTLE_MS = 1_000;

export const copilotBackgroundStream: Scenario = {
  name: "copilot-background-stream",
  description:
    "Start a reply in the 200-message chat, go to Home, and sit there while it keeps streaming.",
  readyWhen:
    "the first words of the reply are on screen, then the Home link is clicked and the chat is gone from the URL, then 1 s",
  windowMs: 15_000,
  async setup(ctx) {
    const { copilot } = await openLongSession({
      ctx,
      paceMs: BACKGROUND_PACE_MS,
    });
    await sendMessage(copilot, "go");
    await ctx.page
      .getByText("Sure. Here is")
      .waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    await ctx.page.getByRole("link", { name: "Home", exact: true }).click();
    await ctx.page.waitForURL((url) => !url.searchParams.has("sessionId"), {
      timeout: READY_TIMEOUT_MS,
    });
    await ctx.page.waitForTimeout(SETTLE_MS);
    return noInteraction;
  },
};
