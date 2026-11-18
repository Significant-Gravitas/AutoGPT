import type { Locator, Page, WebSocketRoute } from "@playwright/test";
import {
  clickRunButton,
  getActiveItemId,
  waitForAgentPageLoad,
} from "../pages/library.page";
import { expect, test } from "./fixtures";

const COUNTDOWN_TIMER_BLOCK_ID = "d67a9c52-5e4e-11e2-bcfd-0800200c9a71";
// Longer than the test's own timeout, so the run never ends by itself: it
// ends only when the journey stops it, after its updates stop arriving.
const RUN_SECONDS = 120;

test.describe("realtime updates journey", () => {
  test("a run that ends while the server's event stream is lost shows as ended once the page reconnects, without a reload", async ({
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

    try {
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

      // The server loses its event stream, so the run's updates stop
      // reaching the page. Only then does the run end.
      live.loseEventStream();
      await stopRun(page, agent.graphID, runID);
      expect(await runStatus(page, agent.graphID, runID)).toBe("TERMINATED");
      await expect
        .poll(() => live.droppedStatuses(runID), {
          message: "the page should have missed the update that ended the run",
        })
        .toContain("TERMINATED");
      // So both views still show it in flight: nothing else updates them.
      await expect(
        runItem.getByRole("img", { name: /^Task (queued|running)$/ }),
        "the runs list should not know the run ended yet",
      ).toBeVisible();
      await expect(activityItem(activity, agent.name)).toContainText(
        /queued|running/,
      );

      // Then the server closes the socket, as it does whenever it loses the
      // stream (#16865), and the page reconnects by itself.
      const reads = countListReads(page, agent.graphID);
      const connections = live.opened();
      await live.closeLikeTheServer();
      await expect
        .poll(() => live.opened(), {
          message: "the page should open a new connection",
          timeout: 30_000,
        })
        .toBeGreaterThan(connections);
      await expect
        .poll(() => reads.runs, {
          message: "the runs list should be read again after the reconnect",
        })
        .toBeGreaterThan(0);
      await expect
        .poll(() => reads.activity, {
          message: "the Activity list should be read again after the reconnect",
        })
        .toBeGreaterThan(0);

      await expect(
        runItem.getByRole("img", { name: "Task stopped" }),
        "the runs list should show the run ended",
      ).toBeVisible();
      await expect(
        activityItem(activity, agent.name),
        "the Activity list should show the run ended",
      ).toContainText(/Stopped/);
      expect(
        live.forwardedStatuses(runID),
        "no update for the run came over the new connection",
      ).toEqual([]);
      expect(await markPageInstance(page), "the page never reloaded").toBe(
        pageInstance,
      );
    } finally {
      // A step that failed before the stop must not leave the run going.
      await stopRun(page, agent.graphID, runID).catch(() => undefined);
    }
  });
});

/**
 * Routes the page's run-updates socket through Playwright, so the journey
 * can break the server side the way production sees it: updates stop
 * arriving, then the server closes the socket with 1011.
 */
async function routeLiveConnection(page: Page) {
  const open: { client: WebSocketRoute; server: WebSocketRoute }[] = [];
  const dropped: { runID: string; status: string }[] = [];
  const forwarded: { runID: string; status: string }[] = [];
  let opened = 0;
  let streamLost = false;

  await page.routeWebSocket(/\/ws(\?|$)/, (client) => {
    const server = client.connectToServer();
    opened += 1;
    open.push({ client, server });
    server.onMessage((message) => {
      const update = runUpdate(message);
      if (!streamLost || !update) {
        client.send(message);
        if (update?.status) forwarded.push(withStatus(update));
      } else if (update.status) {
        dropped.push(withStatus(update));
      }
    });
  });

  return {
    opened: () => opened,
    droppedStatuses: (runID: string) => statusesOf(dropped, runID),
    /** Forwarded from the next time the stream is lost on. */
    forwardedStatuses: (runID: string) => statusesOf(forwarded, runID),
    loseEventStream() {
      streamLost = true;
      forwarded.length = 0;
    },
    async closeLikeTheServer() {
      streamLost = false;
      // The page's side first, so it sees 1011 rather than a normal close
      // forwarded from the server's side.
      for (const { client, server } of open.splice(0)) {
        await client
          .close({ code: 1011, reason: "event stream lost" })
          .catch(() => undefined);
        await server.close().catch(() => undefined);
      }
    },
  };
}

function withStatus(update: { runID: string; status?: string }) {
  return { runID: update.runID, status: String(update.status) };
}

function statusesOf(updates: { runID: string; status: string }[], id: string) {
  return updates
    .filter((update) => update.runID === id)
    .map((update) => update.status);
}

/** Counts the page's reads of the runs list and the Activity list from now on. */
function countListReads(page: Page, graphID: string) {
  const reads = { runs: 0, activity: 0 };
  page.on("request", (request) => {
    if (request.method() !== "GET") return;
    const { pathname } = new URL(request.url());
    if (pathname === `/api/proxy/api/graphs/${graphID}/executions`) {
      reads.runs += 1;
    }
    if (pathname === "/api/proxy/api/executions") reads.activity += 1;
  });
  return reads;
}

/** A run's update from the server, or null for anything else. */
function runUpdate(message: string | Buffer) {
  if (typeof message !== "string") return null;
  try {
    const { method, data } = JSON.parse(message) as {
      method?: string;
      data?: { id?: string; graph_exec_id?: string; status?: string };
    };
    if (method === "graph_execution_event") {
      return { runID: String(data?.id), status: data?.status };
    }
    if (method === "node_execution_event") {
      return { runID: String(data?.graph_exec_id), status: undefined };
    }
    return null;
  } catch {
    return null;
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
        description: "Counts down until stopped, for the realtime journey",
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

/** Returns once the run has ended; the server waits for that. */
async function stopRun(page: Page, graphID: string, runID: string) {
  const response = await page.request.post(
    `/api/proxy/api/graphs/${graphID}/executions/${runID}/stop`,
  );
  expect(response.ok(), `stopping the run: ${response.status()}`).toBe(true);
}

async function runStatus(page: Page, graphID: string, runID: string) {
  const response = await page.request.get(
    `/api/proxy/api/graphs/${graphID}/executions/${runID}`,
  );
  expect(response.ok(), `GET execution: ${response.status()}`).toBe(true);
  return ((await response.json()) as { status: string }).status;
}
