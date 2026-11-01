import { BuildPage } from "../../../src/playwright/pages/build.page";
import { buildGraph } from "../lib/builder-fixture";
import { fulfillJSON, installRoutes } from "../lib/routes";
import { settle, type Scenario } from "../lib/scenario";

const SMALL_GRAPH_ID = "00000000-0000-4000-8000-0000000000c1";
const LARGE_GRAPH_ID = "00000000-0000-4000-8000-0000000000c2";
const LARGE_NODE_COUNT = 100;
const READY_TIMEOUT_MS = 60_000;

export const builderLoadGraph: Scenario = {
  name: "builder-load-graph",
  description: `From an open 1-node workflow, switch the builder to a saved ${LARGE_NODE_COUNT}-node workflow (client-side URL change) and load it.`,
  readyWhen: "the 1-node workflow is on the canvas, then network idle + 1 s",
  windowMs: 2_000,
  async setup(ctx) {
    const { page } = ctx;
    const small = { ...buildGraph(ctx.userID, 1), id: SMALL_GRAPH_ID };
    const large = {
      ...buildGraph(ctx.userID, LARGE_NODE_COUNT),
      id: LARGE_GRAPH_ID,
      name: `Perf: ${LARGE_NODE_COUNT}-step workflow`,
    };
    const graphs = new Map([
      [small.id, small],
      [large.id, large],
    ]);
    await installRoutes(page, {
      graph: async (route, match) => {
        const graph = graphs.get(match[1]);
        if (!graph || route.request().method() !== "GET") {
          return route.fallback();
        }
        await fulfillJSON(route, graph);
      },
    });
    const builder = new BuildPage(page);
    await page.goto(`/build?flowID=${small.id}`);
    await builder.closeTutorial();
    const nodes = page.locator('[data-id^="custom-node-"]');
    await nodes
      .first()
      .waitFor({ state: "attached", timeout: READY_TIMEOUT_MS });
    await settle(page);
    return {
      async run() {
        // The app router follows history.pushState, so this is the same
        // client-side navigation as opening another workflow from the app.
        await page.evaluate((url) => {
          window.history.pushState(null, "", url);
        }, `/build?flowID=${large.id}`);
        await nodes
          .nth(LARGE_NODE_COUNT - 1)
          .waitFor({ state: "attached", timeout: READY_TIMEOUT_MS });
      },
    };
  },
};
