import { BuildPage } from "../../../src/playwright/pages/build.page";
import { createBuilderFixture } from "../lib/builder-fixture";
import { installRoutes } from "../lib/routes";
import { settle, type Scenario } from "../lib/scenario";

const MOVES = 20;
const DRAG_MS = 600;
const READY_TIMEOUT_MS = 60_000;
const EDGE_TIMEOUT_MS = 5_000;

export const builderConnectEdge: Scenario = {
  name: "builder-connect-edge",
  description:
    "Connect one new edge (StoreValue node 4 output → Calculator node 5 input `a`) in an idle 50-node workflow.",
  readyWhen:
    "all 50 nodes are on the canvas and both handles are visible, then network idle + 1 s",
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
    const [source, target] = [fixture.graph.nodes[3], fixture.graph.nodes[4]];
    const sourceHandle = page.locator(
      `[data-tutorial-id="output-handler-${source.id}-output"]`,
    );
    const targetHandle = page.locator(
      `[data-tutorial-id="input-handler-${target.id}-a"]`,
    );
    const newEdge = page.locator(
      `[data-testid="rf__edge-${source.id}:output->${target.id}:a"]`,
    );
    await targetHandle.waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    await settle(page);
    const from = await sourceHandle.boundingBox();
    const to = await targetHandle.boundingBox();
    if (!from || !to) throw new Error("Edge handles have no bounding box");
    return {
      async run() {
        const startX = from.x + from.width / 2;
        const startY = from.y + from.height / 2;
        const endX = to.x + to.width / 2;
        const endY = to.y + to.height / 2;
        await page.mouse.move(startX, startY);
        await page.mouse.down();
        for (let move = 1; move <= MOVES; move++) {
          const progress = move / MOVES;
          await page.mouse.move(
            startX + (endX - startX) * progress,
            startY + (endY - startY) * progress,
          );
          await page.waitForTimeout(DRAG_MS / MOVES);
        }
        await page.mouse.up();
        // A missed or rejected drop would otherwise save a run with no edge write.
        await newEdge.waitFor({ state: "attached", timeout: EDGE_TIMEOUT_MS });
      },
    };
  },
};
