import type { Page, WebSocketRoute } from "@playwright/test";
import {
  createLibraryAgentFixture,
  type LibraryAgentRun,
} from "../lib/library-agent-fixture";
import { installRoutes } from "../lib/routes";
import { settle, type Scenario } from "../lib/scenario";

const READY_TIMEOUT_MS = 60_000;
const SUBSCRIBE_TIMEOUT_MS = 15_000;
const NEW_RUNS = 3;
// Longer than the 2 s execution-event throttle, so every event refetches.
const NEW_RUN_INTERVAL_MS = 2_500;

/**
 * Stands in for the websocket server: acknowledges every subscription and
 * lets the scenario push `graph_execution_event`s at the page.
 */
async function mockExecutionSocket(page: Page) {
  const sockets = new Set<WebSocketRoute>();
  const subscribedGraphIDs = new Set<string>();
  await page.routeWebSocket(/\/ws(\?|$)/, (ws) => {
    sockets.add(ws);
    ws.onClose(() => sockets.delete(ws));
    ws.onMessage((raw) => {
      const message = JSON.parse(String(raw));
      if (message.method === "heartbeat") {
        ws.send(JSON.stringify({ method: "heartbeat", data: "pong" }));
        return;
      }
      if (message.method === "subscribe_graph_executions") {
        subscribedGraphIDs.add(String(message.data?.graph_id));
      }
      ws.send(
        JSON.stringify({
          method: message.method,
          success: true,
          channel: message.data?.graph_id ?? "",
        }),
      );
    });
  });
  function push(run: LibraryAgentRun) {
    const frame = JSON.stringify({
      method: "graph_execution_event",
      success: true,
      data: { ...run, outputs: {}, event_type: "graph_exec_update" },
    });
    for (const ws of sockets) ws.send(frame);
  }
  async function waitForSubscription(graphID: string) {
    const deadline = Date.now() + SUBSCRIBE_TIMEOUT_MS;
    while (!subscribedGraphIDs.has(graphID)) {
      if (Date.now() > deadline) {
        throw new Error(
          `The page never subscribed to ${graphID}; events would reach no one.`,
        );
      }
      await page.waitForTimeout(100);
    }
  }
  return { push, waitForSubscription };
}

export const libraryAgentRuns: Scenario = {
  name: "library-agent-runs",
  description:
    "A workflow's page with 20 runs, the newest RUNNING and selected (it polls every 1.5 s): 3 new runs arrive at the top, 2.5 s apart.",
  readyWhen:
    "the selected run's detail and the 20th run are on screen, network idle + 1 s, and the socket has a subscription for the workflow",
  windowMs: 1_000,
  async setup(ctx) {
    const { page } = ctx;
    const fixture = createLibraryAgentFixture(ctx.userID);
    const socket = await mockExecutionSocket(page);
    await installRoutes(page, fixture.handlers);
    // With both params already in the URL the page does not rewrite it on
    // load, which would otherwise hold back the websocket token action.
    await page.goto(
      `/library/agents/${fixture.agent.id}?activeTab=runs&activeItem=${fixture.selectedRun.id}`,
    );
    const lastRun = fixture.runs[fixture.runs.length - 1];
    await page
      .getByTestId(`library-run-item-${lastRun.id}`)
      .waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    await page
      .getByRole("button", { name: "Output" })
      .first()
      .waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    await settle(page);
    await socket.waitForSubscription(fixture.agent.graph_id);
    return {
      async run() {
        const detailStart = fixture.stats.detailRequests;
        const listStart = fixture.stats.listRequests;
        for (let index = 0; index < NEW_RUNS; index += 1) {
          const run = fixture.addRun();
          socket.push(run);
          await page
            .getByTestId(`library-run-item-${run.id}`)
            .waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
          await page.waitForTimeout(NEW_RUN_INTERVAL_MS);
        }
        console.log(
          `[library-agent-runs] ${NEW_RUNS} new runs: ` +
            `${fixture.stats.detailRequests - detailStart} run-detail GETs, ` +
            `${fixture.stats.listRequests - listStart} run-list GETs`,
        );
      },
    };
  },
};
