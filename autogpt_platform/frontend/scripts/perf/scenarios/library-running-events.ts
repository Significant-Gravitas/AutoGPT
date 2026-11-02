import type { Page, WebSocketRoute } from "@playwright/test";
import { createLibraryFixture } from "../lib/library-fixture";
import { installRoutes, type RouteHandler } from "../lib/routes";
import { settle, type Scenario } from "../lib/scenario";

const READY_TIMEOUT_MS = 60_000;
const EVENT_INTERVAL_MS = 250;
const EVENT_COUNT = 80;
// The fleet summary refetches at most once per 2 s, so a trailing refetch
// can land up to 2 s after the last event; it counts towards the run.
const TRAILING_WINDOW_MS = 2_000;
const SUBSCRIBE_TIMEOUT_MS = 15_000;

type Fixture = ReturnType<typeof createLibraryFixture>;

/**
 * Stands in for the websocket server: answers heartbeats and subscriptions,
 * counts `subscribe_graph_executions` frames, and lets the scenario push
 * `graph_execution_event`s at the page.
 */
async function mockExecutionSocket(page: Page) {
  const sockets = new Set<WebSocketRoute>();
  const stats = { subscribeFrames: 0, subscribedGraphIDs: new Set<string>() };
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
        stats.subscribeFrames += 1;
        stats.subscribedGraphIDs.add(String(message.data?.graph_id));
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
  function push(execution: Fixture["executions"][number]) {
    const frame = JSON.stringify({
      method: "graph_execution_event",
      success: true,
      data: { ...execution, outputs: {}, event_type: "graph_exec_update" },
    });
    for (const ws of sockets) ws.send(frame);
  }
  async function waitForSubscriptions(graphIDs: string[]) {
    const deadline = Date.now() + SUBSCRIBE_TIMEOUT_MS;
    while (!graphIDs.every((id) => stats.subscribedGraphIDs.has(id))) {
      if (Date.now() > deadline) {
        const missing = graphIDs.filter(
          (id) => !stats.subscribedGraphIDs.has(id),
        ).length;
        throw new Error(
          `The page never subscribed to ${missing} of ${graphIDs.length} workflows; events would reach no one.`,
        );
      }
      await page.waitForTimeout(100);
    }
  }
  return { stats, push, waitForSubscriptions };
}

function countRequests(handler: RouteHandler | undefined) {
  const stats = { count: 0 };
  const counted: RouteHandler = async (route, match) => {
    if (route.request().method() === "GET") stats.count += 1;
    await handler?.(route, match);
  };
  return { stats, counted };
}

export const libraryRunningEvents: Scenario = {
  name: "library-running-events",
  description:
    "The library with 20 RUNNING workflows while the websocket pushes an execution event every 250 ms.",
  readyWhen:
    "the 20th workflow is listed, network idle + 1 s, and the socket has a subscription for all 20 workflows",
  windowMs: 0,
  async setup(ctx) {
    const { page } = ctx;
    const fixture = createLibraryFixture(ctx.userID);
    const executions = countRequests(fixture.handlers.executions);
    const socket = await mockExecutionSocket(page);
    await installRoutes(page, {
      ...fixture.handlers,
      executions: executions.counted,
    });
    // With the sort already in the URL the page does not replace it on load;
    // that navigation holds back the websocket token server action, so on a
    // bare /library the socket never opens within the run.
    await page.goto("/library?sort=updatedAt");
    await page
      .getByText(fixture.agents[fixture.agents.length - 1].name)
      .first()
      .waitFor({ state: "visible", timeout: READY_TIMEOUT_MS });
    await settle(page);
    await socket.waitForSubscriptions(
      fixture.agents.map((agent) => agent.graph_id),
    );
    return {
      async run() {
        const subscribesBefore = socket.stats.subscribeFrames;
        const fetchesBefore = executions.stats.count;
        for (let index = 0; index < EVENT_COUNT; index += 1) {
          socket.push(fixture.executions[index % fixture.executions.length]);
          await page.waitForTimeout(EVENT_INTERVAL_MS);
        }
        await page.waitForTimeout(TRAILING_WINDOW_MS);
        const seconds = (EVENT_COUNT * EVENT_INTERVAL_MS) / 1000;
        console.log(
          `[library-running-events] ${EVENT_COUNT} events in ${seconds}s (+ ${TRAILING_WINDOW_MS / 1000}s after): ` +
            `${socket.stats.subscribeFrames - subscribesBefore} subscribe_graph_executions frames, ` +
            `${executions.stats.count - fetchesBefore} GET /executions ` +
            `(setup: ${subscribesBefore} subscribes, ${fetchesBefore} fetches)`,
        );
      },
    };
  },
};
