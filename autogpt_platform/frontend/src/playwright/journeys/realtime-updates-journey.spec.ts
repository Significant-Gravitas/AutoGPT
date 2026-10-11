import { randomBytes } from "crypto";
import type { Locator, Page, Route, WebSocketRoute } from "@playwright/test";
import { expect, test } from "../coverage-fixture";
import { createUniqueAgentName } from "../pages/build.page";
import { clickRunButton, LibraryPage } from "../pages/library.page";

const COUNTDOWN_TIMER_BLOCK_ID = "d67a9c52-5e4e-11e2-bcfd-0800200c9a71";
// Long enough that the connection drops while the run is still going.
const RUN_SECONDS = 10;

test.use({
  storageState: { cookies: [], origins: [] },
  // PostHog's edge serves no flags to "HeadlessChrome", so a preview would
  // show its flag-off layout instead of what users get. Override to choose.
  userAgent:
    process.env.PLAYWRIGHT_JOURNEY_USER_AGENT ??
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
});

test.describe("realtime updates journey", () => {
  test("a run's completion reaches the run view and the Activity menu after the live connection drops, without a reload", async ({
    page,
  }) => {
    test.setTimeout(90_000);
    const liveConnection = await proxyLiveConnection(page);
    await signInJourneyUser(page);
    const agentName = createUniqueAgentName("Journey realtime");
    await createCountdownAgent(page, agentName);

    const library = new LibraryPage(page);
    await library.openSavedAgent(agentName);
    await clickRunButton(page);
    await expect(page.getByRole("img", { name: "Task running" })).toBeVisible({
      timeout: 20_000,
    });
    await expectActivity(page, agentName, /running/i);
    const pageInstance = await markPageInstance(page);

    // The connection drops mid-run, and the first token fetches for the
    // reconnect fail too, as they do while a laptop's network comes back.
    await liveConnection.dropWithFailingTokenFetches(2);

    await expect(page.getByRole("img", { name: "Task completed" })).toBeVisible(
      { timeout: 45_000 },
    );
    await expect.poll(() => library.getRunStatus()).toBe("completed");
    await expectActivity(page, agentName, /completed/i);

    expect(liveConnection.opened()).toBeGreaterThan(1);
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

  await page.routeWebSocket(/\/ws(\?|$)/, (client) => {
    const server = client.connectToServer();
    opened += 1;
    open.push({ client, server });
  });

  return {
    opened: () => opened,
    async dropWithFailingTokenFetches(failures: number) {
      let remaining = failures;
      // getWebSocketToken is a server action: a POST with a Next-Action header.
      await page.route("**/*", (route: Route) => {
        const request = route.request();
        if (
          remaining > 0 &&
          request.method() === "POST" &&
          request.headers()["next-action"]
        ) {
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
  await page.getByTestId("profile-popout-menu-trigger").click();
  await page.getByTestId("account-menu-activity-trigger").click();
}

/** Survives client-side navigation, but not a reload. */
async function markPageInstance(page: Page): Promise<string> {
  return page.evaluate(() => {
    const w = window as Window & { __journeyPageInstance?: string };
    w.__journeyPageInstance ??= crypto.randomUUID();
    return w.__journeyPageInstance;
  });
}

// Onboarding a journey user has already done; the later steps dismiss
// first-visit cards that would cover the page.
const COMPLETED_ONBOARDING_STEPS = [
  "ONBOARDING_COMPLETE",
  "AGENT_NEW_RUN",
  "CAPABILITY_CARDS",
  "AGENTS_TAB_INTRO",
  "MARKETPLACE_TAB_INTRO",
  "BUILD_TAB_INTRO",
  "WORKFLOWS_MOVED",
];

/**
 * Signs up a fresh user, or signs in as PLAYWRIGHT_JOURNEY_EMAIL /
 * PLAYWRIGHT_JOURNEY_PASSWORD when set (e.g. a subscribed user on a preview,
 * where fresh users meet the paywall).
 */
async function signInJourneyUser(page: Page) {
  const headers = { Origin: appOrigin() };
  const email = process.env.PLAYWRIGHT_JOURNEY_EMAIL;
  const password = process.env.PLAYWRIGHT_JOURNEY_PASSWORD;
  if (email && password) {
    const signIn = await page.request.post("/api/auth/sign-in/email", {
      headers,
      data: { email, password },
    });
    expect(signIn.ok(), `sign-in failed: ${signIn.status()}`).toBe(true);
  } else {
    const id = `${Date.now()}-${randomBytes(3).toString("hex")}`;
    const signUp = await page.request.post("/api/auth/sign-up/email", {
      headers,
      data: {
        email: `journey-${id}@${journeyEmailDomain()}`,
        password: `Journey-${randomBytes(12).toString("hex")}`,
        name: `Journey ${id}`,
      },
    });
    expect(signUp.ok(), `sign-up failed: ${signUp.status()}`).toBe(true);
  }
  for (const step of COMPLETED_ONBOARDING_STEPS) {
    const response = await page.request.post("/api/proxy/api/onboarding/step", {
      params: { step },
    });
    expect(response.ok(), `onboarding step ${step} failed`).toBe(true);
  }
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

function appOrigin() {
  return new URL(process.env.PLAYWRIGHT_BASE_URL ?? "http://localhost:3000")
    .origin;
}

// Previews only accept sign-ups from allowlisted domains; @previews.agpt.co
// keeps password sign-up and needs no inbox.
function journeyEmailDomain() {
  if (process.env.PLAYWRIGHT_JOURNEY_EMAIL_DOMAIN) {
    return process.env.PLAYWRIGHT_JOURNEY_EMAIL_DOMAIN;
  }
  const { hostname } = new URL(appOrigin());
  return ["localhost", "127.0.0.1", "[::1]"].includes(hostname)
    ? "example.com"
    : "previews.agpt.co";
}
