import {
  LONG_THREAD_SESSIONS,
  SESSION_S,
  SESSION_XL,
} from "../lib/copilot-fixture";
import { openLongSession } from "../lib/copilot-page";
import { settle, type Scenario } from "../lib/scenario";

const READY_TIMEOUT_MS = 15_000;
const OPEN_TIMEOUT_MS = 120_000;

export const copilotOpenLongThread: Scenario = {
  name: "copilot-open-long-thread",
  description:
    "Open a 2,000-message chat from the sidebar, coming from the 10-message one.",
  readyWhen:
    "the 10-message chat is open and the 2,000-message chat is listed in the sidebar, then network idle + 1 s",
  windowMs: 2_000,
  async setup(ctx) {
    const { page } = ctx;
    await openLongSession({
      ctx,
      sessionID: SESSION_S,
      sessions: LONG_THREAD_SESSIONS,
    });
    const title = LONG_THREAD_SESSIONS[1].title;
    const row = page
      .getByRole("link", { name: title, exact: true })
      .or(page.getByRole("button", { name: title }))
      .first();
    await row.waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    await settle(page);
    return {
      async run() {
        await row.click();
        await page
          .locator(`[data-message-id^="${SESSION_XL}"]`)
          .last()
          .waitFor({ state: "visible", timeout: OPEN_TIMEOUT_MS });
      },
    };
  },
};
