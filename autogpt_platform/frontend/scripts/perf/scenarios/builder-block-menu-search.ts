import { BuildPage } from "../../../src/playwright/pages/build.page";
import { createBuilderFixture } from "../lib/builder-fixture";
import { installRoutes } from "../lib/routes";
import { settle, type Scenario } from "../lib/scenario";

const QUERY = "text";
const KEY_DELAY_MS = 120;
const SCROLLS = 6;
const SCROLL_PX = 800;
const SCROLL_PAUSE_MS = 400;
const READY_TIMEOUT_MS = 60_000;

export const builderBlockMenuSearch: Scenario = {
  name: "builder-block-menu-search",
  description: `Open the block menu over an idle 50-node workflow, type "${QUERY}" into its search, then scroll the results ${SCROLLS} times so more pages load.`,
  readyWhen:
    "all 50 nodes are on the canvas and the block menu is open on its default view, then network idle + 1 s",
  windowMs: 2_000,
  async setup(ctx) {
    const { page } = ctx;
    const fixture = createBuilderFixture(ctx.userID);
    await installRoutes(page, fixture.handlers);
    const builder = new BuildPage(page);
    await page.goto(`/build?flowID=${fixture.graph.id}`);
    await builder.closeTutorial();
    await page
      .locator('[data-id^="custom-node-"]')
      .nth(fixture.graph.nodes.length - 1)
      .waitFor({ state: "attached", timeout: READY_TIMEOUT_MS });
    await builder.openBlocksPanel();
    const search = page.locator(
      '[data-id="blocks-control-search-bar"] input[type="text"]',
    );
    await search.waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    await settle(page);
    const menu = page.locator('[data-id="blocks-control-popover-content"]');
    return {
      async run() {
        await search.click();
        await page.keyboard.type(QUERY, { delay: KEY_DELAY_MS });
        await menu
          .locator('[data-id^="block-card-"]')
          .first()
          .waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
        const box = await menu.boundingBox();
        if (!box) throw new Error("Block menu has no bounding box");
        await page.mouse.move(box.x + box.width * 0.75, box.y + box.height / 2);
        for (let scroll = 0; scroll < SCROLLS; scroll++) {
          await page.mouse.wheel(0, SCROLL_PX);
          await page.waitForTimeout(SCROLL_PAUSE_MS);
        }
      },
    };
  },
};
