import type { Page, WebSocketRoute } from "@playwright/test";
import { BuildPage } from "../../../src/playwright/pages/build.page";
import { createBuilderFixture } from "../lib/builder-fixture";
import { installRoutes } from "../lib/routes";
import { settle, type Scenario } from "../lib/scenario";

const FIRST_PENDING_NODE = 30;
const EVENT_INTERVAL_MS = 50;
const READY_TIMEOUT_MS = 60_000;
const FINISHED_AT = "2026-01-05T09:00:05.000Z";

type Execution = ReturnType<typeof createBuilderFixture>["execution"];
type NodeExecution = Execution["node_executions"][number];

/**
 * The 10 nodes still RUNNING or QUEUED go RUNNING, then COMPLETED. Every event
 * arrives twice, as it does when the poll and the websocket both deliver it.
 */
function nodeEvents(execution: Execution) {
  return execution.node_executions
    .slice(FIRST_PENDING_NODE)
    .flatMap((nodeExecution: NodeExecution) => {
      const running = {
        ...nodeExecution,
        status: "RUNNING",
        start_time: nodeExecution.queue_time,
      };
      const completed = {
        ...running,
        status: "COMPLETED",
        output_data: { output: [nodeExecution.node_exec_id] },
        end_time: FINISHED_AT,
      };
      return [running, running, completed, completed];
    });
}

/**
 * Passes the builder's websocket through to the real server, except the
 * subscribe for the fixture run, which the harness acknowledges itself (the
 * backend has no such run and would reject it).
 */
export async function routeExecutionSocket(page: Page, executionID: string) {
  let socket: WebSocketRoute | null = null;
  let markSubscribed: () => void = () => undefined;
  const subscribed = new Promise<void>((resolve) => {
    markSubscribed = resolve;
  });
  await page.routeWebSocket(/\/ws(\?|$)/, (ws) => {
    const server = ws.connectToServer();
    socket = ws;
    ws.onMessage((message) => {
      const parsed = JSON.parse(String(message));
      if (
        parsed.method === "subscribe_graph_execution" &&
        parsed.data?.graph_exec_id === executionID
      ) {
        ws.send(
          JSON.stringify({
            method: "subscribe_graph_execution",
            success: true,
            channel: `perf|graph_exec#${executionID}`,
          }),
        );
        markSubscribed();
        return;
      }
      server.send(message);
    });
  });
  return {
    subscribed,
    send(method: string, data: unknown) {
      socket?.send(JSON.stringify({ method, data }));
    },
  };
}

export async function withTimeout<T>(
  promise: Promise<T>,
  ms: number,
  what: string,
) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error(`Timed out waiting for ${what}`)),
          ms,
        );
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

export const builderRunEvents: Scenario = {
  name: "builder-run-events",
  description:
    "A 50-node workflow's RUNNING run receives 40 node execution events over the websocket (10 nodes go RUNNING then COMPLETED, every event twice), one every 50 ms.",
  readyWhen:
    "all 50 nodes are on the canvas with the running execution loaded and the websocket subscribe acknowledged, then network idle + 1 s",
  windowMs: 2_000,
  async setup(ctx) {
    const { page } = ctx;
    const fixture = createBuilderFixture(ctx.userID);
    await installRoutes(page, fixture.handlers);
    const socket = await routeExecutionSocket(page, fixture.execution.id);
    const builder = new BuildPage(page);
    await page.goto(
      `/build?flowID=${fixture.graph.id}&flowExecutionID=${fixture.execution.id}`,
    );
    await builder.closeTutorial();
    await page
      .locator('[data-id^="custom-node-"]')
      .nth(fixture.graph.nodes.length - 1)
      .waitFor({ state: "attached", timeout: READY_TIMEOUT_MS });
    await withTimeout(
      socket.subscribed,
      READY_TIMEOUT_MS,
      "websocket subscribe",
    );
    await settle(page);
    const { node_executions } = fixture.execution;
    const events = nodeEvents(fixture.execution);
    return {
      async run() {
        for (const event of events) {
          // The poll sees what the websocket has sent, as with the real backend.
          const index = node_executions.findIndex(
            (nodeExecution) =>
              nodeExecution.node_exec_id === event.node_exec_id,
          );
          node_executions[index] = event;
          socket.send("node_execution_event", event);
          await page.waitForTimeout(EVENT_INTERVAL_MS);
        }
      },
    };
  },
};
