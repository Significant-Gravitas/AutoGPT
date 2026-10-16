import type { BrowserContext, Page, Request } from "@playwright/test";
import { CopilotPage } from "../pages/copilot.page";
import { expect, test } from "./fixtures";
import quietThenAnswerScript from "./model-scripts/quiet-then-answer.json";
import {
  expectEachTurnOnce,
  type ModelScript,
  readJournal,
  requireScriptedModel,
  scriptedRun,
} from "./scripted-model";

// A copilot turn keeps exactly one stream while what can resume it fires: a
// tab wake, and the stall watchdog over a quiet stretch carrying only
// heartbeats. Two streams on one chat crashed the AI SDK's request cleanup
// (Sentry BUILDER-777) and dropped the turn's end handling; the watchdog read
// heartbeat-only stretches as a dead stream and showed "Connection lost".
// A connection dropped mid-answer is copilot-reconnect-journey.spec.ts.
//
// The model goes quiet for 9 s before it answers: long enough for the
// stream registry's heartbeat (after 5 s of silence), short of the silence
// watchdog's status (10 s).

const SCRIPT: ModelScript = quietThenAnswerScript;
const ANSWER = SCRIPT.turns[0].text ?? "";
const FOLLOW_UP = SCRIPT.turns[1].expect_user_text ?? "";
const FOLLOW_UP_ANSWER = SCRIPT.turns[1].text ?? "";

test.beforeAll(requireScriptedModel);

test("copilot resume journey: one stream per turn through a tab wake and a heartbeat-only stretch", async ({
  page,
  context,
  browserName,
  journeyUser: _user,
}) => {
  test.skip(browserName !== "chromium", "reads stream bytes through CDP");
  test.setTimeout(90_000);

  await watchConnectionLostToasts(page);
  const streams = watchStreams(page);
  const bytes = await watchStreamBytes(context, page);
  const run = scriptedRun("quiet-then-answer");

  // A tab wake and the stall watchdog while the turn is quiet, on a fake
  // clock.
  await page.clock.install();
  const copilot = new CopilotPage(page);
  await openNewChat(copilot);
  bytes.markTurn();
  await send(page, copilot, `Think it over. ${run.marker}`);
  await setTabVisibility(page, "hidden");
  await page.clock.fastForward(31_000);
  await setTabVisibility(page, "visible");
  await bytes.waitForHeartbeat();
  // Past the 60 s stall watchdog, counted from the last visible change but
  // not from the heartbeat.
  await page.clock.fastForward(30_000);

  const transcript = page.getByRole("log");
  await expect(
    transcript.getByText(ANSWER, { exact: true }),
    "the answer should show once",
  ).toHaveCount(1, { timeout: 30_000 });
  expect(streams.resumes, "a healthy stream is never resumed beside").toBe(0);
  expect(await connectionLostToasts(page)).toBe(0);

  // The chat still takes a follow-up, whose turn streams and ends.
  await waitForTurnOver(page, copilot);
  await send(page, copilot, FOLLOW_UP);
  await expect(
    transcript.getByText(FOLLOW_UP_ANSWER, { exact: true }),
    "the follow-up's answer should show once",
  ).toHaveCount(1, { timeout: 30_000 });

  expect(streams.maxOpen, "streams open at once on one chat").toBe(1);
  expectEachTurnOnce(await readJournal(run.nonce), SCRIPT);
});

async function waitForTurnOver(page: Page, copilot: CopilotPage) {
  await expect(page.getByRole("button", { name: /stop/i })).toHaveCount(0, {
    timeout: 15_000,
  });
  const sessionId = new URL(page.url()).searchParams.get("sessionId");
  await expect
    .poll(
      async () => {
        const response = await page.request.get(
          `/api/proxy/api/chat/sessions/${sessionId}`,
        );
        return (await response.json()).active_stream ?? null;
      },
      { timeout: 15_000 },
    )
    .toBeNull();
  await expect(copilot.getChatInput()).toBeEnabled();
}

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
