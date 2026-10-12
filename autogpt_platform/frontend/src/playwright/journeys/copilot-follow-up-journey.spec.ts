import type { Page } from "@playwright/test";
import {
  currentSessionID,
  firstWords,
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
  requireScriptedModel,
  scriptedRun,
} from "./scripted-model";

// A slow first answer, then the answer to the follow-up the script expects
// the model to be shown.
const SCRIPT: ModelScript = followUpScript;
const [ANSWER, FOLLOW_UP_ANSWER] = SCRIPT.turns.map((turn) => turn.text ?? "");
const FOLLOW_UP = SCRIPT.turns[1].expect_user_text ?? "";

test.describe("copilot follow-up journey", () => {
  test.beforeAll(requireScriptedModel);

  test("a follow-up queued while AutoPilot answers runs once, after the answer", async ({
    page,
    journeyUser,
  }) => {
    const run = scriptedRun("answer-then-follow-up");
    const firstMessage = `Tell me about today. ${run.marker}`;
    await startChat(page, firstMessage);
    const transcript = page.getByRole("log");
    await expect(
      transcript.getByText(firstWords(ANSWER)),
      "the answer should start streaming",
    ).toBeVisible();

    await page
      .getByRole("textbox", { name: "Chat message input" })
      .fill(FOLLOW_UP);
    await page.getByRole("button", { name: "Queue message" }).click();

    await expect(
      transcript.getByText(FOLLOW_UP_ANSWER, { exact: true }),
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

    const journal = await readJournal(run.nonce);
    expectEachTurnOnce(journal, SCRIPT);
    if (!journeyUser.id) throw new Error("the journey user should have an id");
    await expectChargedWhatWasBilled(journeyUser.id, journal);
  });
});

async function savedConversation(page: Page, sessionID: string) {
  return (await savedMessages(page, sessionID))
    .filter((message) => ["user", "assistant"].includes(message.role))
    .map((message) => [message.role, message.content]);
}
