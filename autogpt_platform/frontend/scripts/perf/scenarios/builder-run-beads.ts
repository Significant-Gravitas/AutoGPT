import { BuildPage } from "../../../src/playwright/pages/build.page";
import { createBuilderFixture } from "../lib/builder-fixture";
import { installRoutes } from "../lib/routes";
import { settle, type Scenario } from "../lib/scenario";
import { routeExecutionSocket, withTimeout } from "./builder-run-events";

const FIRST_RERUN_NODE = 30;
const EVENT_INTERVAL_MS = 150;
const READY_TIMEOUT_MS = 60_000;
const FINISHED_AT = "2026-01-05T09:00:05.000Z";

type Execution = ReturnType<typeof createBuilderFixture>["execution"];

/**
 * A second, finished execution of each of the 10 nodes from node 30 on. Each
 * has a new node_exec_id and input data, so every edge into a StoreValue node
 * among them gets a new bead that travels the edge and is removed on arrival.
 */
function rerunEvents(execution: Execution) {
  return execution.node_executions
    .slice(FIRST_RERUN_NODE)
    .map((nodeExecution, index) => ({
      ...nodeExecution,
      node_exec_id: `00000000-0000-4000-b000-${String(index).padStart(12, "0")}`,
      status: "COMPLETED",
      start_time: nodeExecution.queue_time,
      end_time: FINISHED_AT,
      output_data: { output: [`rerun ${index}`] },
    }));
}

export const builderRunBeads: Scenario = {
  name: "builder-run-beads",
  description:
    "A 50-node workflow's RUNNING run receives 10 new node executions over the websocket, one every 150 ms; each sends a bead along the edge into its node.",
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
    const events = rerunEvents(fixture.execution);
    return {
      async run() {
        for (const event of events) {
          // The poll sees what the websocket has sent, as with the real backend.
          fixture.execution.node_executions.push(event);
          socket.send("node_execution_event", event);
          await page.waitForTimeout(EVENT_INTERVAL_MS);
        }
      },
    };
  },
};
