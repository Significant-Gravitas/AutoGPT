import type {
  BrowserContext,
  Page,
  Request as PlaywrightRequest,
  Route,
} from "@playwright/test";
import {
  currentSessionID,
  cutChatStreams,
  firstWords,
  isChatStream,
  recordChatStreams,
  savedAssistantTexts,
  savedMessages,
  startChat,
} from "./copilot-chat";
import { expect, test } from "./fixtures";
import type { JourneyUser } from "./journey-user";
import droppedSendScript from "./model-scripts/dropped-send.json";
import slowAnswerScript from "./model-scripts/slow-answer.json";
import {
  expectChargedWhatWasBilled,
  expectEachTurnOnce,
  type ModelScript,
  readJournal,
  releaseHeldReply,
  requireScriptedModel,
  scriptedRun,
} from "./scripted-model";

// One answer, streamed a word at a time. The model sends its first
// HELD_AFTER_WORDS words, then holds the rest until the journey has cut the
// connection, so the journey waits for fewer words than that to show.
const SCRIPT: ModelScript = slowAnswerScript;
const ANSWER = SCRIPT.turns[0].text ?? "";
const HELD_AFTER_WORDS = slowAnswerScript.turns[0].stream.hold_after_words;
const STREAMING_WORDS = Math.floor(HELD_AFTER_WORDS / 2);

const DROPPED_SEND_SCRIPT: ModelScript = droppedSendScript;
const DROPPED_SEND_ANSWER = DROPPED_SEND_SCRIPT.turns[0].text ?? "";
// A send, not a resume: a resume's URL carries its turn and cursor.
const CHAT_SEND = /\/api\/chat\/sessions\/[^/]+\/stream$/;

test.describe("copilot reconnect journey", () => {
  // A resume after the turn finished finds its cursor trimmed, and the chat
  // seeds from the checkpoint this 409 carries.
  test.use({
    expectedResponses: [
      {
        status: 409,
        method: "GET",
        url: /\/api\/chat\/sessions\/[^/]+\/stream\?turn=/,
      },
    ],
  });
  test.beforeAll(requireScriptedModel);

  test("an answer cut off by a dropped connection arrives whole, once and as saved", async ({
    page,
    context,
    journeyUser,
  }) => {
    await loseConnectionMidAnswer({
      page,
      context,
      journeyUser,
      offline: false,
    });
  });

  test("an answer cut off while the browser is offline arrives whole, once and as saved", async ({
    page,
    context,
    journeyUser,
  }) => {
    await loseConnectionMidAnswer({
      page,
      context,
      journeyUser,
      offline: true,
    });
  });
});

test.describe("copilot reconnect journey, a dropped send", () => {
  test.beforeAll(requireScriptedModel);

  test("a message the network drops before it leaves goes again on Send again, once", async ({
    page,
    journeyUser,
  }) => {
    const run = scriptedRun("dropped-send");
    const message = `Answer me once. ${run.marker}`;
    // The first send fails as it does offline, before it reaches the server;
    // later requests go through untouched, a CORS preflight included.
    async function dropFirstSend(route: Route) {
      if (route.request().method() !== "POST") return route.fallback();
      await route.abort("internetdisconnected");
      await page.unroute(CHAT_SEND, dropFirstSend);
    }
    await page.route(CHAT_SEND, dropFirstSend);
    const dropped = page.waitForRequest(isChatStream("POST"));
    void dropped.catch(() => undefined);
    await startChat(page, message);
    const firstID = messageIDOf(await dropped);
    expect(firstID, "the send should name its message id").toBeTruthy();

    // The chat reads the session, finds no row for the message, and says so.
    const banner = page.getByTestId("chat-error-banner");
    await expect(
      banner.getByText(
        "The connection dropped before your message was sent. Check your connection and send it again.",
      ),
      "the chat should say the message didn't go",
    ).toBeVisible({ timeout: 30_000 });
    const resent = page.waitForRequest(isChatStream("POST"));
    void resent.catch(() => undefined);
    await banner.getByRole("button", { name: "Send again" }).click();
    expect(
      messageIDOf(await resent),
      "Send again should post the message under its first id",
    ).toBe(firstID);

    await expect(
      page.getByRole("log").getByText(DROPPED_SEND_ANSWER, { exact: true }),
      "the answer should show, once",
    ).toHaveCount(1);
    await expect(banner, "the banner should go once it's sent").toHaveCount(0);
    const sessionID = await currentSessionID(page);
    await expect
      .poll(() => savedConversation(page, sessionID), {
        message: "the chat should save the message and its answer, once each",
      })
      .toEqual([
        ["user", message],
        ["assistant", DROPPED_SEND_ANSWER],
      ]);
    await page.reload();
    await expect(
      page.getByRole("log").getByText(DROPPED_SEND_ANSWER, { exact: true }),
      "the reloaded chat should show the answer once",
    ).toHaveCount(1);

    expectEachTurnOnce(await readJournal(run.nonce), DROPPED_SEND_SCRIPT);
    if (!journeyUser.id) throw new Error("the journey user should have an id");
    await expectChargedWhatWasBilled(journeyUser.id, [run.nonce]);
  });
});

function messageIDOf(request: PlaywrightRequest) {
  return (request.postDataJSON() as { message_id?: string }).message_id;
}

async function savedConversation(page: Page, sessionID: string) {
  return (await savedMessages(page, sessionID))
    .filter((message) => ["user", "assistant"].includes(message.role))
    .map((message) => [message.role, message.content]);
}

interface Args {
  page: Page;
  context: BrowserContext;
  journeyUser: JourneyUser;
  offline: boolean;
}

async function loseConnectionMidAnswer({
  page,
  context,
  journeyUser,
  offline,
}: Args) {
  const run = scriptedRun("slow-answer");
  await recordChatStreams(page);
  const transcript = page.getByRole("log");
  let reconnected: ReturnType<Page["waitForRequest"]>;
  let released = false;
  try {
    await startChat(page, `Answer me slowly. ${run.marker}`);
    await expect(
      transcript.getByText(firstWords(ANSWER, STREAMING_WORDS)),
      "the answer should start streaming",
    ).toBeVisible({ timeout: 60_000 });

    // The connection drops mid-answer while the model is still writing it.
    // Online, one drop resumes silently: the chat speaks up from the second
    // failed attempt in a row.
    reconnected = page.waitForRequest(isChatStream("GET"));
    // A failure before it is awaited mustn't leave it to reject unhandled
    // when the page closes; the success path still awaits it below.
    void reconnected.catch(() => undefined);
    if (offline) await context.setOffline(true);
    await cutChatStreams(page);
    // Only now does the model write the rest, while the connection is down:
    // the cut can't miss an answer that finished before the first words
    // showed.
    await releaseHeldReply(run.nonce);
    released = true;
  } finally {
    // A journey that fails before the cut must not leave the reply held open
    // on the scripted model. Each release lets one held reply through, so
    // this one goes out only if the journey's own didn't.
    if (!released) {
      await releaseHeldReply(run.nonce).catch(() => undefined);
      if (offline) await context.setOffline(false).catch(() => undefined);
    }
  }
  if (offline) {
    await expect(
      page.getByText("You're offline", { exact: true }),
      "the page should say the browser is offline",
    ).toBeVisible();
    await expect(
      transcript.getByText(/Offline — waiting for the network/).first(),
      "the chat should say it is waiting for the network",
    ).toBeVisible();
    await context.setOffline(false);
  }
  await reconnected;

  await expect(
    transcript.getByText(ANSWER, { exact: true }),
    "the whole answer should show, once, without a reload",
  ).toHaveCount(1);
  const sessionID = await currentSessionID(page);
  await expect
    .poll(() => savedAssistantTexts(page, sessionID), {
      message: "the chat should save the answer once",
    })
    .toEqual([ANSWER]);
  await page.reload();
  await expect(
    page.getByRole("log").getByText(ANSWER, { exact: true }),
    "the reloaded chat should show the answer once",
  ).toHaveCount(1);

  const journal = await readJournal(run.nonce);
  expectEachTurnOnce(journal, SCRIPT);
  if (!journeyUser.id) throw new Error("the journey user should have an id");
  await expectChargedWhatWasBilled(journeyUser.id, [run.nonce]);
}
