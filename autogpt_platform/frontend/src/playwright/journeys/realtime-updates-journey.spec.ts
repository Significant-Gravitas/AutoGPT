import type {
  Locator,
  Page,
  Request,
  Route,
  WebSocketRoute,
} from "@playwright/test";
import { createUniqueAgentName } from "../pages/build.page";
import { clickRunButton, LibraryPage } from "../pages/library.page";
import { expect, test } from "./fixtures";

const COUNTDOWN_TIMER_BLOCK_ID = "d67a9c52-5e4e-11e2-bcfd-0800200c9a71";
// Long enough that the connection drops while the run is still going.
const RUN_SECONDS = 20;

test.describe("realtime updates journey", () => {
  test("a run's completion reaches the run view and the Activity menu after the live connection drops, without a reload", async ({
    page,
    journeyUser: _user,
  }) => {
    // A shared CI executor can keep a run queued for a while.
    test.setTimeout(180_000);
    const liveConnection = await proxyLiveConnection(page);
    const agentName = createUniqueAgentName("Journey realtime");
    await createCountdownAgent(page, agentName);

    const library = new LibraryPage(page);
    await library.openSavedAgent(agentName);
    // Land on the agent page with a full load, as from a link or a refresh,
    // so the page sets up its live connection from scratch.
    const openedBeforeReload = liveConnection.opened();
    await page.reload();
    await expect
      .poll(() => liveConnection.opened(), {
        message: "the page never opened its live connection",
        timeout: 30_000,
      })
      .toBeGreaterThan(openedBeforeReload);
    await clickRunButton(page);
    const pageInstance = await markPageInstance(page);
    // In flight (queued or running) right before the drop, so the completion
    // has to cross it.
    await expect(
      page.getByRole("img", { name: /^Task (queued|running)$/ }),
    ).toBeVisible({ timeout: 30_000 });

    // The connection drops mid-run, and the first token fetches for the
    // reconnect fail too, as they do while a laptop's network comes back.
    await liveConnection.dropWithFailingTokenFetches(2);

    await expect(page.getByRole("img", { name: "Task completed" })).toBeVisible(
      { timeout: 90_000 },
    );
    await expect.poll(() => library.getRunStatus()).toBe("completed");
    await expectActivity(page, agentName, /completed/i);

    expect(liveConnection.opened()).toBeGreaterThan(openedBeforeReload + 1);
    expect(await markPageInstance(page)).toBe(pageInstance);
  });
});

/**
 * Routes the app's WebSocket through Playwright so the test can cut it the
 * way a network blip or a server restart does: both ends see it close.
 */
async function proxyLiveConnection(page: Page) {
  const open: { client: WebSocketRoute; server: WebSocketRoute }[] = [];
  let opened = 0;

  // The client's own account of its connection, in the test output.
  page.on("console", (message) => {
    if (/BackendAPI|WebSocket|Heartbeat/.test(message.text())) {
      console.log(`[browser] ${message.text()}`);
    }
  });

  await page.routeWebSocket(/\/ws(\?|$)/, (client) => {
    const server = client.connectToServer();
    opened += 1;
    open.push({ client, server });
  });

  return {
    opened: () => opened,
    async dropWithFailingTokenFetches(failures: number) {
      let remaining = failures;
      await page.route("**/*", (route: Route) => {
        if (remaining > 0 && isTokenFetch(route.request())) {
          remaining -= 1;
          return route.abort("internetdisconnected");
        }
        return route.fallback();
      });
      for (const { client, server } of open.splice(0)) {
        await server.close({ code: 1001, reason: "journey: connection lost" });
        await client.close({ code: 1001, reason: "journey: connection lost" });
      }
    },
  };
}

/** The client's token fetch: the token route, or (on older builds) the
 * getWebSocketToken server action, a POST with a Next-Action header. */
function isTokenFetch(request: Request): boolean {
  if (new URL(request.url()).pathname === "/api/auth/token") return true;
  return request.method() === "POST" && !!request.headers()["next-action"];
}

/** Opens the Activity menu and checks the agent's latest run there. */
async function expectActivity(page: Page, agentName: string, status: RegExp) {
  const menu = page.getByTestId("agent-activity-dropdown");
  await openActivityMenu(page);
  await expect(menu).toBeVisible();
  const item: Locator = menu
    .getByRole("button")
    .filter({ hasText: agentName })
    .first();
  await expect(item).toContainText(status, { timeout: 20_000 });
  await page.keyboard.press("Escape");
  await page.keyboard.press("Escape");
  await expect(menu).toBeHidden();
}

// The classic layout has an Activity button in the navbar; the new layout
// keeps the same list under the account menu's Activity row.
async function openActivityMenu(page: Page) {
  const navbarButton = page.getByTestId("agent-activity-button");
  if (await navbarButton.isVisible()) {
    await navbarButton.click();
    return;
  }
  const activityRow = page.getByTestId("account-menu-activity-trigger");
  if (!(await activityRow.isVisible())) {
    await page.getByTestId("profile-popout-menu-trigger").click();
  }
  await activityRow.click();
}

/** Survives client-side navigation, but not a reload. */
async function markPageInstance(page: Page): Promise<string> {
  return page.evaluate(() => {
    const w = window as Window & { __journeyPageInstance?: string };
    w.__journeyPageInstance ??= crypto.randomUUID();
    return w.__journeyPageInstance;
  });
}

async function createCountdownAgent(page: Page, name: string) {
  const response = await page.request.post("/api/proxy/api/graphs", {
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
  expect(response.ok(), `creating the agent failed: ${response.status()}`).toBe(
    true,
  );
}
