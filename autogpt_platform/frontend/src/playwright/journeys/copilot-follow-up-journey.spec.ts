import type { Page, Request as PlaywrightRequest } from "@playwright/test";
import {
  currentSessionID,
  firstWords,
  hasActiveTurn,
  savedMessages,
  startChat,
} from "./copilot-chat";
import { expect, test } from "./fixtures";
import followUpScript from "./model-scripts/answer-then-follow-up.json";
import {
  expectChargedWhatWasBilled,
  expectEachTurnOnce,
  type ModelScript,
  readJournal,
  releaseHeldReply,
  requireScriptedModel,
  scriptedRun,
} from "./scripted-model";

// A first answer, then the answer to the follow-up the script expects the
// model to be shown. The model sends the first answer's first
// HELD_AFTER_WORDS words and holds the rest until the journey releases it, so
// the follow-up is sent while the answer is still being written, however fast
// the model or slow the page.
const SCRIPT: ModelScript = followUpScript;
const [ANSWER, FOLLOW_UP_ANSWER] = SCRIPT.turns.map((turn) => turn.text ?? "");
const FOLLOW_UP = SCRIPT.turns[1].expect_user_text ?? "";
const HELD_AFTER_WORDS = followUpScript.turns[0].stream?.hold_after_words ?? 0;
const STREAMING_WORDS = Math.floor(HELD_AFTER_WORDS / 2);
// POST queues a follow-up behind the running turn; GET peeks at the queue.
const QUEUE_URL = /\/api\/chat\/sessions\/[^/]+\/messages\/pending$/;

type Run = ReturnType<typeof scriptedRun>;

test.describe("copilot follow-up journey", () => {
  test.beforeAll(requireScriptedModel);

  test("a follow-up queued while AutoPilot answers runs once, after the answer", async ({
    page,
    journeyUser,
  }) => {
    const run = scriptedRun("answer-then-follow-up");
    const firstMessage = `Tell me about today. ${run.marker}`;
    await whileTheAnswerIsHeld(page, run, firstMessage, async () => {
      const queued = page.waitForResponse((response) =>
        isQueuePost(response.request()),
      );
      void queued.catch(() => undefined);
      await queueFollowUp(page);
      expect(
        (await queued).status(),
        "the running turn should take the follow-up",
      ).toBe(200);
    });
    await expectFollowUpAnsweredOnce(page, run, firstMessage, journeyUser.id);
  });
});

test.describe("copilot follow-up journey, as the answer ends", () => {
  // The follow-up reaches the queue only once the answer's turn has ended, so
  // the queue refuses it (#17247), and the chat sends it as a turn of its own.
  // By then the turn is marked completed, so the refusal is the status
  // check's: this covers how the client handles a refused follow-up. The
  // window between the last look and the turn's end is pinned by
  // backend/copilot/sdk/follow_up_after_last_look_test.py.
  test.use({
    expectedResponses: [{ status: 409, method: "POST", url: QUEUE_URL }],
  });
  test.beforeAll(requireScriptedModel);

  test("a follow-up the queue refuses as the answer ends still runs once, after the answer", async ({
    page,
    journeyUser,
  }) => {
    const run = scriptedRun("answer-then-follow-up");
    const firstMessage = `Tell me about today. ${run.marker}`;
    const letQueueRequestThrough = await holdQueueRequests(page);
    try {
      await whileTheAnswerIsHeld(page, run, firstMessage, async () => {
        const sent = page.waitForRequest(isQueuePost);
        void sent.catch(() => undefined);
        await queueFollowUp(page);
        await sent;
      });
      const sessionID = await currentSessionID(page);
      // As long as the answer may take to start: an SDK turn has stalled
      // 15 s between the model's last message and the end of its stream.
      await expect
        .poll(() => hasActiveTurn(page, sessionID), {
          message: "the answer's turn should end while the follow-up waits",
          timeout: 60_000,
        })
        .toBe(false);
      const refused = page.waitForResponse((response) =>
        isQueuePost(response.request()),
      );
      void refused.catch(() => undefined);
      letQueueRequestThrough();
      expect(
        (await refused).status(),
        "the queue should refuse a follow-up no turn will read",
      ).toBe(409);
    } finally {
      letQueueRequestThrough();
    }
    await expectFollowUpAnsweredOnce(page, run, firstMessage, journeyUser.id);
  });
});

/**
 * Opens the chat, runs `act` while the first answer is held part-way through,
 * then lets the answer finish.
 */
async function whileTheAnswerIsHeld(
  page: Page,
  run: Run,
  firstMessage: string,
  act: () => Promise<void>,
) {
  expect(STREAMING_WORDS, "the script should hold the answer").toBeGreaterThan(
    0,
  );
  let released = false;
  try {
    await startChat(page, firstMessage);
    await expect(
      page.getByRole("log").getByText(firstWords(ANSWER, STREAMING_WORDS)),
      "the answer should start streaming",
    ).toBeVisible({ timeout: 60_000 });
    await act();
    await releaseHeldReply(run.nonce);
    released = true;
  } finally {
    // Each release lets one held reply through, so this one goes out only if
    // the journey failed before its own.
    if (!released) await releaseHeldReply(run.nonce).catch(() => undefined);
  }
}

async function queueFollowUp(page: Page) {
  await page
    .getByRole("textbox", { name: "Chat message input" })
    .fill(FOLLOW_UP);
  await page.getByRole("button", { name: "Queue message" }).click();
}

function isQueuePost(request: PlaywrightRequest) {
  return (
    request.method() === "POST" &&
    QUEUE_URL.test(new URL(request.url()).pathname)
  );
}

/**
 * Holds the browser's follow-ups on their way to the queue until the returned
 * function lets them through.
 */
async function holdQueueRequests(page: Page) {
  let letThrough = () => {};
  const released = new Promise<void>((resolve) => {
    letThrough = resolve;
  });
  await page.route(QUEUE_URL, async (route) => {
    if (isQueuePost(route.request())) await released;
    // A journey that failed lets the request go as its page closes.
    await route.fallback().catch(() => undefined);
  });
  return () => letThrough();
}

async function expectFollowUpAnsweredOnce(
  page: Page,
  run: Run,
  firstMessage: string,
  userID: string | undefined,
) {
  await expect(
    page.getByRole("log").getByText(FOLLOW_UP_ANSWER, { exact: true }),
    "the follow-up should be answered, once",
  ).toHaveCount(1);
  const sessionID = await currentSessionID(page);
  await expect
    .poll(() => savedConversation(page, sessionID), {
      message:
        "the chat should save the follow-up once, after the first answer, with its own answer",
    })
    .toEqual([
      ["user", firstMessage],
      ["assistant", ANSWER],
      ["user", FOLLOW_UP],
      ["assistant", FOLLOW_UP_ANSWER],
    ]);
  await page.reload();
  for (const text of [ANSWER, FOLLOW_UP, FOLLOW_UP_ANSWER]) {
    await expect(
      page.getByRole("log").getByText(text, { exact: true }),
      "the reloaded chat should show each message once",
    ).toHaveCount(1);
  }

  expectEachTurnOnce(await readJournal(run.nonce), SCRIPT);
  if (!userID) throw new Error("the journey user should have an id");
  await expectChargedWhatWasBilled(userID, [run.nonce]);
}

async function savedConversation(page: Page, sessionID: string) {
  return (await savedMessages(page, sessionID))
    .filter((message) => ["user", "assistant"].includes(message.role))
    .map((message) => [message.role, message.content]);
}
