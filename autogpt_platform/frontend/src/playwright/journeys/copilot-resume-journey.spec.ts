import type { BrowserContext, Page, Request } from "@playwright/test";
import { CopilotPage } from "../pages/copilot.page";
import { expect, test } from "./fixtures";

// A copilot turn keeps exactly one stream while everything that can resume it
// fires: a network blip, a tab wake, and the stall watchdog over a quiet
// stretch carrying only heartbeats. Two streams on one chat crashed the AI SDK's
// request cleanup (Sentry BUILDER-777) and dropped the turn's end handling;
// the watchdog read heartbeat-only stretches as a dead stream and showed
// "Connection lost".
//
// Needs the backend in CHAT_TEST_MODE: `__test_quiet_pause__` makes the dummy
// go quiet for 9 s, long enough for one heartbeat and short of the 10 s
// silence status.

test.use({
  // Going offline leaves an unrelated effect's fetch rejection unhandled; the
  // AI SDK's BUILDER-777 TypeError is still caught by the guard.
  expectedErrors: [/^Failed to fetch$/],
});

const ANSWER = "I counted: 1... 2... 3. All done!";
// Distinct per turn: the composer drops a repeat of the last message.
const QUIET_PROMPT = "__test_quiet_pause__ count to three";
const BLIP_PROMPT = "__test_quiet_pause__ count again";

test("copilot resume journey: one stream per turn through a network blip, a tab wake and quiet stretches", async ({
  page,
  context,
  browserName,
  journeyUser: _user,
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
  const bytes = await watchStreamBytes(context, page);

  const copilot = new CopilotPage(page);
  const answers = page.getByText(ANSWER, { exact: true });

  // Each scenario gets its own chat: the dummy always gives the same reply,
  // and a chat currently drops a later reply identical to an earlier one.

  // A network blip mid-turn. The turn still completes, on one stream.
  await openNewChat(copilot);
  await send(page, copilot, BLIP_PROMPT);
  await context.setOffline(true);
  await expect.poll(() => page.evaluate(() => navigator.onLine)).toBe(false);
  await context.setOffline(false);
  await expect(answers).toHaveCount(1, { timeout: 30_000 });

  // A tab wake and the stall watchdog while the turn is quiet, on a fake
  // clock from here on.
  await page.clock.install();
  await openNewChat(copilot);
  const resumesBefore = streams.resumes;
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
  expect(
    streams.resumes - resumesBefore,
    "a healthy stream is never resumed beside",
  ).toBe(0);
  expect(await connectionLostToasts(page)).toBe(0);

  // The chat still takes a follow-up, whose turn streams and ends.
  const followUpEnded = page.waitForEvent(
    "requestfinished",
    (request) => isStreamRequest(request) && request.method() === "POST",
  );
  await send(page, copilot, "thanks");
  await followUpEnded;
  await expect(page.getByText("thanks", { exact: true })).toBeVisible();
  await expect(copilot.getChatInput()).toBeEnabled({ timeout: 15_000 });

  expect(streams.maxOpen, "streams open at once on one chat").toBe(1);
});

async function openNewChat(copilot: CopilotPage) {
  const sessionId = await copilot.createSessionViaApi();
  await copilot.open(sessionId);
  await copilot.waitForChatInput();
}

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
