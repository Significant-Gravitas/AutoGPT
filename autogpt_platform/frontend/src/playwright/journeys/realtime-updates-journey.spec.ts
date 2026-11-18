import type { Locator, Page, WebSocketRoute } from "@playwright/test";
import {
  clickRunButton,
  getActiveItemId,
  waitForAgentPageLoad,
} from "../pages/library.page";
import { expect, test } from "./fixtures";

const COUNTDOWN_TIMER_BLOCK_ID = "d67a9c52-5e4e-11e2-bcfd-0800200c9a71";
// Long enough that the run is still going when its updates stop arriving.
const RUN_SECONDS = 8;

test.describe("realtime updates journey", () => {
  test("a run that finishes while the server's event stream is lost shows as finished once the page reconnects, without a reload", async ({
    page,
    journeyUser: _user,
  }) => {
    const live = await routeLiveConnection(page);
    const agent = await createCountdownAgent(page);

    await page.goto(`/library/agents/${agent.libraryAgentID}`);
    await waitForAgentPageLoad(page);
    await clickRunButton(page);
    await expect
      .poll(() => getActiveItemId(page), {
        message: "starting a run should open it in the run view",
      })
      .toBeTruthy();
    const runID = String(getActiveItemId(page));
    const pageInstance = await markPageInstance(page);
    const runItem = page.getByTestId(`library-run-item-${runID}`);
    await expect(
      runItem.getByRole("img", { name: /^Task (queued|running)$/ }),
      "the runs list should show the run in flight",
    ).toBeVisible({ timeout: 30_000 });
    const activity = await openActivity(page);
    await expect(activityItem(activity, agent.name)).toContainText(
      /queued|running/,
    );

    // The server loses its event stream, so the run's updates stop reaching
    // the page, and the run finishes meanwhile.
    live.loseEventStream();
    await expect
      .poll(() => runStatus(page, agent.graphID, runID), {
        message: "the run should finish while its updates are lost",
        timeout: 45_000,
      })
      .toBe("COMPLETED");
    // Then the server closes the socket, as it does whenever it loses the
    // stream (#16865), and the page reconnects by itself.
    const connections = live.opened();
    await live.closeLikeTheServer();
    await expect
      .poll(() => live.opened(), {
        message: "the page should open a new connection",
        timeout: 30_000,
      })
      .toBeGreaterThan(connections);

    await expect(
      runItem.getByRole("img", { name: "Task completed" }),
      "the runs list should show the run finished",
    ).toBeVisible();
    await expect(
      activityItem(activity, agent.name),
      "the Activity list should show the run finished",
    ).toContainText(/Completed/);
    expect(await markPageInstance(page), "the page never reloaded").toBe(
      pageInstance,
    );
  });
});

/**
 * Routes the page's run-updates socket through Playwright, so the journey
 * can break the server side the way production sees it: updates stop
 * arriving, then the server closes the socket with 1011.
 */
async function routeLiveConnection(page: Page) {
  const open: { client: WebSocketRoute; server: WebSocketRoute }[] = [];
  let opened = 0;
  let streamLost = false;

  await page.routeWebSocket(/\/ws(\?|$)/, (client) => {
    const server = client.connectToServer();
    opened += 1;
    open.push({ client, server });
    server.onMessage((message) => {
      if (streamLost && isRunUpdate(message)) return;
      client.send(message);
    });
  });

  return {
    opened: () => opened,
    loseEventStream() {
      streamLost = true;
    },
    async closeLikeTheServer() {
      streamLost = false;
      for (const { client, server } of open.splice(0)) {
        await server.close().catch(() => undefined);
        await client
          .close({ code: 1011, reason: "event stream lost" })
          .catch(() => undefined);
      }
    },
  };
}

function isRunUpdate(message: string | Buffer) {
  if (typeof message !== "string") return false;
  try {
    const { method } = JSON.parse(message) as { method?: string };
    return (
      method === "graph_execution_event" || method === "node_execution_event"
    );
  } catch {
    return false;
  }
}

async function openActivity(page: Page) {
  await page.getByTestId("profile-popout-menu-trigger").click();
  await page.getByTestId("account-menu-activity-trigger").click();
  const activity = page.getByTestId("agent-activity-dropdown");
  await expect(activity).toBeVisible();
  return activity;
}

function activityItem(activity: Locator, agentName: string) {
  return activity.getByRole("link").filter({ hasText: agentName }).first();
}

/** Survives client-side navigation, but not a reload. */
async function markPageInstance(page: Page): Promise<string> {
  return page.evaluate(() => {
    const w = window as Window & { __journeyPageInstance?: string };
    w.__journeyPageInstance ??= crypto.randomUUID();
    return w.__journeyPageInstance;
  });
}

async function createCountdownAgent(page: Page) {
  const name = `Journey realtime ${Date.now()}`;
  const created = await page.request.post("/api/proxy/api/graphs", {
    data: {
      source: "upload",
      graph: {
        name,
        description: "Waits a few seconds, for the realtime updates journey",
        is_active: true,
        nodes: [
          {
            id: "countdown",
            block_id: COUNTDOWN_TIMER_BLOCK_ID,
            input_default: { seconds: RUN_SECONDS, input_message: "done" },
            metadata: { position: { x: 0, y: 0 } },
            input_links: [],
            output_links: [],
          },
        ],
        links: [],
      },
    },
  });
  expect(created.ok(), `creating the agent: ${created.status()}`).toBe(true);
  const graph = (await created.json()) as { id: string };
  const libraryAgent = await page.request.get(
    `/api/proxy/api/library/agents/by-graph/${graph.id}`,
  );
  expect(
    libraryAgent.ok(),
    `finding it in the Library: ${libraryAgent.status()}`,
  ).toBe(true);
  const { id } = (await libraryAgent.json()) as { id: string };
  return { name, graphID: graph.id, libraryAgentID: id };
}

async function runStatus(page: Page, graphID: string, runID: string) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${graphID}/executions/${runID}`,
  );
  expect(response.ok(), `GET execution: ${response.status()}`).toBe(true);
  return ((await response.json()) as { status: string }).status;
}
