import { BuildPage } from "../../../src/playwright/pages/build.page";
import { createBuilderFixture } from "../lib/builder-fixture";
import { installRoutes } from "../lib/routes";
import { settle, type Scenario } from "../lib/scenario";

const MOVES = 60;
const DRAG_MS = 2_000;
const STEP_PX = 4;
const READY_TIMEOUT_MS = 60_000;

export const builderDragRunning: Scenario = {
  name: "builder-drag-running",
  description:
    "Drag one node of a 50-node workflow for 2 s (60 pointer moves) while its run is RUNNING.",
  readyWhen:
    "all 50 nodes are on the canvas with the running execution loaded, then network idle + 1 s",
  windowMs: 2_000,
  async setup(ctx) {
    const { page } = ctx;
    const fixture = createBuilderFixture(ctx.userID);
    await installRoutes(page, fixture.handlers);
    const builder = new BuildPage(page);
    await page.goto(
      `/build?flowID=${fixture.graph.id}&flowExecutionID=${fixture.execution.id}`,
    );
    await builder.closeTutorial();
    const nodes = page.locator('[data-id^="custom-node-"]');
    await nodes
      .nth(fixture.graph.nodes.length - 1)
      .waitFor({ state: "attached", timeout: READY_TIMEOUT_MS });
    await settle(page);
    const box = await nodes.first().boundingBox();
    if (!box) throw new Error("First node has no bounding box");
    return {
      async run() {
        const x = box.x + box.width / 2;
        const y = box.y + 12;
        await page.mouse.move(x, y);
        await page.mouse.down();
        for (let move = 1; move <= MOVES; move++) {
          await page.mouse.move(x + move * STEP_PX, y + move * (STEP_PX / 2));
          await page.waitForTimeout(DRAG_MS / MOVES);
        }
        await page.mouse.up();
      },
    };
  },
};
