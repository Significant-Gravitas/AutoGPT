import { COPILOT_SESSIONS } from "../lib/copilot-fixture";
import { openLongSession } from "../lib/copilot-page";
import { settle, type Scenario } from "../lib/scenario";

const ROUNDS = 3;
const DWELL_MS = 2_000;
const READY_TIMEOUT_MS = 15_000;

export const copilotSwitchThreads: Scenario = {
  name: "copilot-switch-threads",
  description:
    "Switch between chats of 200, 50 and 10 messages from the sidebar, three rounds, 2 s on each.",
  readyWhen:
    "the 200-message chat is open and all three chats are listed in the expanded sidebar, then network idle + 1 s",
  windowMs: 0,
  async setup(ctx) {
    const { page } = ctx;
    await openLongSession({ ctx });
    const expand = page.getByRole("button", { name: "Expand sidebar" });
    if (await expand.isVisible()) await expand.click();
    const loadAll = page.getByRole("button", { name: "Load all Otto chats" });
    if (await loadAll.isVisible()) await loadAll.click();
    // The AppSidebar lists chats as links, the copilot's own ChatSidebar as
    // buttons whose name also carries the date.
    const links = COPILOT_SESSIONS.map((session) =>
      page
        .getByRole("link", { name: session.title, exact: true })
        .or(page.getByRole("button", { name: session.title }))
        .first(),
    );
    for (const link of links) {
      await link.waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    }
    await settle(page);
    const [long, medium, short] = links;
    return {
      async run() {
        for (let round = 0; round < ROUNDS; round++) {
          for (const link of [medium, short, long]) {
            await link.click();
            await page.waitForTimeout(DWELL_MS);
          }
        }
      },
    };
  },
};
