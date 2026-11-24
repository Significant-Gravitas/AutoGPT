import type { Locator, Page, Request, WebSocketRoute } from "@playwright/test";
import { subscribeTimedOutFrame } from "../../tests/integrations/subscribe-failure-frame";
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
// Longer than the runs list's 2 s refresh throttle
// (EXECUTION_REFETCH_THROTTLE_MS), whose last refresh can trail its events.
const LISTS_QUIET_MS = 2_500;

test.describe("realtime updates journey", () => {
  test("a run that ends while the server's event stream is lost shows as ended once the page reconnects, without a reload", async ({
    page,
    journeyUser: _user,
  }) => {
    const live = await routeLiveConnection(page);
    const agent = await createCountdownAgent(page);
    const reads = trackListReads(page, agent.graphID);

    await page.goto(`/library/agents/${agent.libraryAgentID}`);
    await waitForAgentPageLoad(page);
    // A workflow with no runs has no live view on its page, so open the
    // account menu: its Activity row subscribes to every workflow. The server
    // can't complete that subscribe, as when Redis doesn't confirm it in time
    // (#16922), and nothing else subscribes to the workflow before the run.
    live.failNextSubscribe(agent.graphID);
    await page.getByTestId("profile-popout-menu-trigger").click();
    const menu = page.getByTestId("account-menu-popover");
    await expect(menu).toBeVisible();
    await expect
      .poll(() => live.subscribeFailed(), {
        message:
          "the page should subscribe to the workflow, and the server fail that",
        timeout: 15_000,
      })
      .toBe(true);
    await expect
      .poll(() => live.failedSubscribeConfirmedAgain(), {
        message:
          "the page should send the subscribe the server could not complete " +
          "again, or the run's live updates never arrive",
        timeout: 15_000,
      })
      .toBe(true);
    await page.keyboard.press("Escape");
    await expect(menu).toBeHidden();
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
      await expect
        .poll(() => live.forwardedStatuses(runID).length, {
          message: "live update arrived after one failed resubscribe",
          timeout: 30_000,
        })
        .toBeGreaterThan(0);
      const activity = await openActivity(page);
      await expect(activityItem(activity, agent.name)).toContainText(
        /queued|running/,
      );

      // A read already on its way could see the run end, so let the lists'
      // last refreshes land first. Then the server loses its event stream,
      // so the run's updates stop reaching the page. Only then does it end.
      await reads.settled();
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
      const connections = live.opened();
      await live.closeLikeTheServer();
      await expect
        .poll(() => live.opened(), {
          message: "the page should open a new connection",
          timeout: 30_000,
        })
        .toBeGreaterThan(connections);
      const reopenedAt = live.reopenedAt();
      await expect
        .poll(() => reads.runsSince(reopenedAt), {
          message: "the runs list should be read again after the reconnect",
        })
        .toBeGreaterThan(0);
      await expect
        .poll(() => reads.activitySince(reopenedAt), {
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
      expect(
        live.urls().filter((url) => new URL(url).search),
        "every connection sends its token in its first message, not its URL",
      ).toEqual([]);
    } finally {
      // A step that failed before the stop must not leave the run going.
      await stopRun(page, agent.graphID, runID).catch(() => undefined);
    }
  });
});

/**
 * Routes the page's run-updates socket through Playwright, so the journey
 * can break the server side the way production sees it: a subscribe it
 * can't complete, updates that stop arriving, then a close with 1011.
 */
async function routeLiveConnection(page: Page) {
  const connections: Connection[] = [];
  let closedUpTo = 0;
  let outage = false;
  const failure = { graphID: "", failed: false, confirmedAgain: false };

  // What the journey breaks must not reach the real Sentry, whose tunnel is
  // on the app's own origin.
  await page
    .context()
    .route(/sentry\.io|[?&]o=\d+&p=\d+/, (route) =>
      route.fulfill({ status: 200, body: "{}" }),
    );

  await page.routeWebSocket(/\/ws(\?|$)/, (client) => {
    const connection: Connection = {
      client,
      server: client.connectToServer(),
      openedAt: Date.now(),
      // One the page opens during the outage gets no updates either.
      streamLost: outage,
      subscribing: [],
      dropped: [],
      forwarded: [],
    };
    connections.push(connection);
    client.onMessage((message) => {
      const subscribe = subscribeRequest(message);
      if (
        subscribe?.graphID === failure.graphID &&
        subscribe.method === "subscribe_graph_executions" &&
        !failure.failed
      ) {
        failure.failed = true;
        client.send(subscribeTimedOutFrame(subscribe.graphID));
        return;
      }
      if (subscribe) {
        connection.subscribing.push({
          graphID: subscribe.graphID,
          sentAfterFailure: failure.failed,
        });
      }
      connection.server.send(message);
    });
    connection.server.onMessage((message) => {
      // The server answers a socket's subscribes in order.
      const answer = subscribeAnswer(message);
      if (answer) {
        const subscribe = connection.subscribing.shift();
        if (
          subscribe?.sentAfterFailure &&
          subscribe.graphID === failure.graphID &&
          answer === "ok"
        ) {
          failure.confirmedAgain = true;
        }
      }
      const update = runUpdate(message);
      if (connection.streamLost && update) {
        if (update.status) connection.dropped.push(withStatus(update));
        return;
      }
      connection.client.send(message);
      if (update?.status) connection.forwarded.push(withStatus(update));
    });
  });

  return {
    /** Fails the page's next subscribe to the graph's runs, once. */
    failNextSubscribe(graphID: string) {
      failure.graphID = graphID;
    },
    /** The page sent that subscribe, and the server's failure went back. */
    subscribeFailed: () => failure.failed,
    /**
     * The server confirmed a subscribe to the graph sent after the failure.
     * Until Run, only the page's retry can be one: the workflow's page has no
     * live view, and the menu's Activity row subscribes to each workflow once.
     */
    failedSubscribeConfirmedAgain: () => failure.confirmedAgain,
    opened: () => connections.length,
    urls: () => connections.map(({ client }) => client.url()),
    /** When the first connection after the last close opened. */
    reopenedAt: () => connections[closedUpTo]?.openedAt ?? Infinity,
    droppedStatuses: (runID: string) =>
      statusesOf(
        connections.flatMap((connection) => connection.dropped),
        runID,
      ),
    /** Forwarded over the connections opened since the last close. */
    forwardedStatuses: (runID: string) =>
      statusesOf(
        connections.slice(closedUpTo).flatMap((c) => c.forwarded),
        runID,
      ),
    /** Until the server closes them, connections get no run updates. */
    loseEventStream() {
      outage = true;
      for (const connection of connections) connection.streamLost = true;
    },
    async closeLikeTheServer() {
      const closing = connections.slice(closedUpTo);
      closedUpTo = connections.length;
      outage = false;
      // The page's side first, so it sees 1011 rather than a normal close
      // forwarded from the server's side.
      for (const { client, server } of closing) {
        await client
          .close({ code: 1011, reason: "event stream lost" })
          .catch(() => undefined);
        await server.close().catch(() => undefined);
      }
    },
  };
}

interface Connection {
  client: WebSocketRoute;
  server: WebSocketRoute;
  openedAt: number;
  streamLost: boolean;
  /** Each subscribe sent to the server and not yet answered. */
  subscribing: { graphID?: string; sentAfterFailure: boolean }[];
  dropped: { runID: string; status: string }[];
  forwarded: { runID: string; status: string }[];
}

function withStatus(update: { runID: string; status?: string }) {
  return { runID: update.runID, status: String(update.status) };
}

function statusesOf(updates: { runID: string; status: string }[], id: string) {
  return updates
    .filter((update) => update.runID === id)
    .map((update) => update.status);
}

/** Tracks the page's reads of the runs list and the Activity list. */
function trackListReads(page: Page, graphID: string) {
  const runs: number[] = [];
  const activity: number[] = [];
  let inFlight = 0;
  let lastChange = Date.now();

  function listOf(request: Request) {
    if (request.method() !== "GET") return null;
    const { pathname } = new URL(request.url());
    if (pathname === `/api/proxy/api/graphs/${graphID}/executions`) return runs;
    if (pathname === "/api/proxy/api/executions") return activity;
    return null;
  }
  function finished(request: Request) {
    if (!listOf(request)) return;
    inFlight -= 1;
    lastChange = Date.now();
  }
  page.on("request", (request) => {
    const list = listOf(request);
    if (!list) return;
    list.push(Date.now());
    inFlight += 1;
    lastChange = Date.now();
  });
  page.on("requestfinished", finished);
  page.on("requestfailed", finished);

  return {
    runsSince: (time: number) => runs.filter((at) => at >= time).length,
    activitySince: (time: number) => activity.filter((at) => at >= time).length,
    /** Waits until no read is out and none has started for a while. */
    async settled() {
      await expect
        .poll(
          () => inFlight === 0 && Date.now() - lastChange > LISTS_QUIET_MS,
          {
            message: "the lists' refreshes should settle",
            timeout: 15_000,
          },
        )
        .toBe(true);
    },
  };
}

/** A subscribe the page sends, or null for anything else. */
function subscribeRequest(message: string | Buffer) {
  const parsed = parseMessage(message);
  if (!parsed?.method?.startsWith("subscribe_")) return null;
  return { method: parsed.method, graphID: parsed.data?.graph_id };
}

/** The server's answer to a subscribe, or null for anything else. */
function subscribeAnswer(message: string | Buffer) {
  const parsed = parseMessage(message);
  // An error answers a subscribe only if it names one.
  if (parsed?.method === "error") {
    return parsed.request?.method?.startsWith("subscribe_") ? "failed" : null;
  }
  if (parsed?.method?.startsWith("subscribe_")) return "ok";
  return null;
}

function parseMessage(message: string | Buffer) {
  if (typeof message !== "string") return null;
  try {
    return JSON.parse(message) as {
      method?: string;
      data?: { graph_id?: string };
      request?: { method?: string };
    };
  } catch {
    return null;
  }
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
