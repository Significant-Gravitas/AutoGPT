import { createLibraryFixture } from "../lib/library-fixture";
import { installRoutes } from "../lib/routes";
import { noInteraction, settle, type Scenario } from "../lib/scenario";

const READY_TIMEOUT_MS = 60_000;

export const libraryRunning: Scenario = {
  name: "library-running",
  description: "The library with 20 workflows, all of them RUNNING.",
  readyWhen: "the 20th workflow is listed, then network idle + 1 s",
  windowMs: 30_000,
  async setup(ctx) {
    const { page } = ctx;
    const fixture = createLibraryFixture(ctx.userID);
    await installRoutes(page, fixture.handlers);
    await page.goto("/library");
    await page
      .getByText(fixture.agents[fixture.agents.length - 1].name)
      .first()
      .waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    await settle(page);
    return noInteraction;
  },
};
