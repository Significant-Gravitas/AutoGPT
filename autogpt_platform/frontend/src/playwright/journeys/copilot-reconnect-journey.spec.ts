import type { BrowserContext, Page } from "@playwright/test";
import {
  currentSessionID,
  cutChatStreams,
  firstWords,
  isChatStream,
  recordChatStreams,
  savedAssistantTexts,
  startChat,
} from "./copilot-chat";
import { expect, test } from "./fixtures";
import type { JourneyUser } from "./journey-user";
import slowAnswerScript from "./model-scripts/slow-answer.json";
import {
  expectChargedWhatWasBilled,
  expectEachTurnOnce,
  type ModelScript,
  readJournal,
  requireScriptedModel,
  scriptedRun,
} from "./scripted-model";

// One answer, streamed a word at a time over a few seconds.
const SCRIPT: ModelScript = slowAnswerScript;
const ANSWER = SCRIPT.turns[0].text ?? "";

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
  await startChat(page, `Answer me slowly. ${run.marker}`);
  const transcript = page.getByRole("log");
  await expect(
    transcript.getByText(firstWords(ANSWER)),
    "the answer should start streaming",
  ).toBeVisible();

  // The connection drops mid-answer while the model is still writing it.
  // Online, one drop resumes silently: the chat speaks up from the second
  // failed attempt in a row.
  const reconnected = page.waitForRequest(isChatStream("GET"));
  if (offline) await context.setOffline(true);
  await cutChatStreams(page);
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
  await expectChargedWhatWasBilled(journeyUser.id, journal);
}
