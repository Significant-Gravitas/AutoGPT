import {
  expect,
  test,
  type BrowserContext,
  type Page,
  type Request,
} from "@playwright/test";
import { CopilotPage } from "../pages/copilot.page";
import { signupTestUser } from "../utils/signup";

// A copilot turn keeps exactly one stream while everything that can resume it
// fires: a tab wake, the stall watchdog over a quiet stretch carrying only
// heartbeats, and a network blip. Two streams on one chat crashed the AI SDK's
// request cleanup (Sentry BUILDER-777) and dropped the turn's end handling;
// the watchdog read heartbeat-only stretches as a dead stream and showed
// "Connection lost".
//
// Needs the backend in CHAT_TEST_MODE: `__test_quiet_pause__` makes the dummy
// go quiet for 9 s, long enough for one heartbeat and short of the 10 s
// silence status.

// PostHog answers a HeadlessChrome user agent with every flag off.
test.use({
  userAgent:
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
});

const ANSWER = "I counted: 1... 2... 3. All done!";
// Distinct per turn: the composer drops a repeat of the last message.
const QUIET_PROMPT = "__test_quiet_pause__ count to three";
const BLIP_PROMPT = "__test_quiet_pause__ count again";

test("copilot resume journey: one stream per turn through wake, quiet stretches and a network blip", async ({
  page,
  context,
  browserName,
}) => {
  test.skip(browserName !== "chromium", "reads stream bytes through CDP");
  test.skip(
    new URL(
      test.info().project.use.baseURL ?? "http://localhost",
    ).hostname.endsWith(".vercel.app"),
    "previews run real models; this journey needs CHAT_TEST_MODE",
  );
  test.setTimeout(90_000);

  await watchConnectionLostToasts(page);
  const streams = watchStreams(page);
  const crashes = watchCrashes(page);
  const bytes = await watchStreamBytes(context, page);

  await signupTestUser(page, journeyEmail(), undefined, true);
  await page.clock.install();
  const copilot = new CopilotPage(page);
  const sessionId = await copilot.createSessionViaApi();
  await copilot.open(sessionId);
  await copilot.waitForChatInput();
  const answers = page.getByText(ANSWER, { exact: true });

  // Turn 1: wake and watchdog while the turn is quiet.
  bytes.markTurn();
  await send(page, copilot, QUIET_PROMPT);
  await setTabVisibility(page, "hidden");
  await page.clock.fastForward(31_000);
  await setTabVisibility(page, "visible");
  await bytes.waitForHeartbeat();
  // Past the 60 s stall watchdog, counted from the last visible change but
  // not from the heartbeat.
  await page.clock.fastForward(30_000);

  await expect(answers).toHaveCount(1, { timeout: 20_000 });
  expect(streams.resumes, "a healthy stream is never resumed beside").toBe(0);
  expect(await connectionLostToasts(page)).toBe(0);

  // Turn 2: a network blip mid-turn.
  await send(page, copilot, BLIP_PROMPT);
  await context.setOffline(true);
  await expect.poll(() => page.evaluate(() => navigator.onLine)).toBe(false);
  await context.setOffline(false);
  // A blip can leave the stream open but dead. Past the stall watchdog, the
  // chat recovers it by replacing it, never by running a second one.
  await page.clock.fastForward(61_000);
  await expect(answers).toHaveCount(2, { timeout: 30_000 });

  // Turn 3: the chat still takes a follow-up.
  await send(page, copilot, "thanks");
  await expect(answers).toHaveCount(3, { timeout: 20_000 });

  expect(streams.maxOpen, "streams open at once on one chat").toBe(1);
  expect(crashes).toEqual([]);
});

async function send(page: Page, copilot: CopilotPage, text: string) {
  const posted = page.waitForResponse(
    (response) =>
      isStreamRequest(response.request()) &&
      response.request().method() === "POST",
  );
  const input = copilot.getChatInput();
  await expect(input).toBeEnabled({ timeout: 15_000 });
  await input.fill(text);
  await input.press("Enter");
  await posted;
}

function isStreamRequest(request: Request) {
  return /\/api\/chat\/sessions\/[^/]+\/stream(\?|$)/.test(request.url());
}

/** Stream requests open at once, and how many were resumes (GET). */
function watchStreams(page: Page) {
  const open = new Set<Request>();
  const state = { maxOpen: 0, resumes: 0 };
  page.on("request", (request) => {
    if (!isStreamRequest(request)) return;
    if (request.method() === "GET") state.resumes += 1;
    open.add(request);
    state.maxOpen = Math.max(state.maxOpen, open.size);
  });
  const close = (request: Request) => open.delete(request);
  page.on("requestfinished", close);
  page.on("requestfailed", close);
  return state;
}

/** Page errors, and the AI SDK's own report of BUILDER-777. */
function watchCrashes(page: Page) {
  const crashes: string[] = [];
  page.on("pageerror", (error) => crashes.push(error.message));
  page.on("console", (message) => {
    if (message.type() !== "error") return;
    if (message.text().includes("reading 'state'")) {
      crashes.push(message.text());
    }
  });
  return crashes;
}

/**
 * Bytes arriving on stream responses, read through CDP: Playwright only
 * exposes a streaming body once it has ended.
 */
async function watchStreamBytes(context: BrowserContext, page: Page) {
  const cdp = await context.newCDPSession(page);
  await cdp.send("Network.enable");
  const streamIds = new Set<string>();
  const arrivals: number[] = [];
  cdp.on("Network.requestWillBeSent", (event) => {
    if (/\/api\/chat\/sessions\/[^/]+\/stream(\?|$)/.test(event.request.url)) {
      streamIds.add(event.requestId);
    }
  });
  cdp.on("Network.dataReceived", (event) => {
    if (streamIds.has(event.requestId)) arrivals.push(Date.now());
  });
  let turnStart = 0;
  return {
    markTurn() {
      turnStart = arrivals.length;
    },
    /** Wait for bytes after a quiet gap in this turn: the registry's heartbeat. */
    async waitForHeartbeat() {
      await expect
        .poll(
          () =>
            arrivals.some(
              (at, i) => i > turnStart && at - arrivals[i - 1] > 3000,
            ),
          { timeout: 15_000 },
        )
        .toBe(true);
    },
  };
}

/** Toasts leave after a few seconds, so record every one as it mounts. */
async function watchConnectionLostToasts(page: Page) {
  await page.addInitScript(() => {
    const seen: string[] = [];
    Object.assign(window, { __connectionLostToasts: seen });
    new MutationObserver((records) => {
      for (const record of records) {
        record.addedNodes.forEach((node) => {
          if (node.textContent?.includes("Connection lost")) {
            seen.push(node.textContent);
          }
        });
      }
    }).observe(document, { childList: true, subtree: true });
  });
}

function connectionLostToasts(page: Page) {
  return page.evaluate(
    () =>
      (window as unknown as { __connectionLostToasts: string[] })
        .__connectionLostToasts.length,
  );
}

async function setTabVisibility(page: Page, state: "hidden" | "visible") {
  await page.evaluate((value) => {
    Object.defineProperty(document, "visibilityState", {
      value,
      configurable: true,
    });
    Object.defineProperty(document, "hidden", {
      value: value === "hidden",
      configurable: true,
    });
    document.dispatchEvent(new Event("visibilitychange"));
  }, state);
}

function journeyEmail() {
  const domain = process.env.PLAYWRIGHT_JOURNEY_EMAIL_DOMAIN;
  return domain ? `journey-resume-${Date.now()}@${domain}` : undefined;
}
