import type { Locator, Page, Response } from "@playwright/test";
import {
  chatStreamCount,
  currentSessionID,
  expectChatSettled,
  finishedChatStreams,
  recordChatStreams,
  savedMessages,
  startChat,
} from "./copilot-chat";
import {
  type Credential,
  deleteCredential,
  expectListedInIntegrations,
  fakeAPIKey,
  listCredentials,
  openIntegrations,
} from "./credentials";
import { expect, test } from "./fixtures";
import connectScript from "./model-scripts/connect-card-api-key.json";
import cancelScript from "./model-scripts/connect-card-cancelled.json";
import {
  expectChargedWhatWasBilled,
  expectEachTurnOnce,
  type ModelScript,
  readJournal,
  readJournalForKey,
  requireScriptedModel,
  scriptedRun,
  streamedTexts,
} from "./scripted-model";

// AutoPilot runs the AI Text Generator with Claude. The stack has no platform
// LLM keys, so the block needs the user's own Anthropic key and answers with a
// sign-in card, which the chat shows as one connect card under the step. The
// journeys stack points Anthropic clients at the scripted model, which
// journals which key each request presented.
const CONNECT: ModelScript = connectScript;
const CANCEL: ModelScript = cancelScript;
const PROVIDER = "anthropic";
const SERVICE = "Anthropic";
const FIRST_MESSAGE = "Write me one line about the sea with Claude.";
// What the card sends once the key is saved (SetupRequirementsCard helpers).
const CONNECTED = CONNECT.turns[2].expect_user_text ?? "";
const SKIP = CANCEL.turns[2].expect_user_text ?? "";
const BLOCK_MODEL = "claude-haiku-4-5-20251001";
const CARD_HEADING = /^(Connect your apps|Your apps are connected)$/;
const STILL_WAITING = "Once these apps are connected, Otto will continue.";
// A model turn starts a Claude Agent SDK session, which takes a while on CI.
const MODEL_TURN_TIMEOUT = 60_000;
// On dev the card is back as soon as the chat renders. A known failure must
// end as a failed assertion, not a timeout, so this stays well under the
// test's budget.
const REDO_CHECK_TIMEOUT = 5_000;

test.describe("copilot connect card journey", () => {
  test.beforeAll(requireScriptedModel);

  test("Otto asks for an API key on one connect card, the user adds it there, and the chat carries on by itself and runs the step once with that key", async ({
    page,
    journeyUser,
  }) => {
    const run = scriptedRun("connect-card-api-key");
    const key = fakeAPIKey();
    const keyName = `Journey key ${run.nonce}`;
    const texts = CONNECT.turns.map((turn) => turn.text ?? "");

    await recordChatStreams(page);
    const sessionID = await askForTheLine(page, run.marker, CONNECT);
    const credential = await connectKeyOnTheCard(page, { key, keyName });
    try {
      const transcript = page.getByRole("log");
      await expect(
        transcript.getByText(texts[3], { exact: true }),
        "the chat should carry on by itself and finish, once",
      ).toHaveCount(1, { timeout: MODEL_TURN_TIMEOUT });
      await expect(
        connectCards(transcript),
        "the card should go once it has sent its follow-up",
      ).toHaveCount(0);
      const streamed = (await finishedChatStreams(page, "POST", 2)).flatMap(
        streamedTexts,
      );
      expect(streamed, "the streams should carry what the model said").toEqual(
        texts,
      );
      expect(
        await chatStreamCount(page, "POST"),
        "the card should send one follow-up, and nothing else should",
      ).toBe(2);
      await expectChatSettled(page, sessionID);

      expect(
        await savedKeys(page),
        "the key should be saved once, as the user named it",
      ).toEqual([[keyName, "api_key"]]);
      expect(
        (await readJournalForKey(key)).map((call) => [
          call.kind,
          call.model,
          call.completed,
        ]),
        "the step should call the provider once, with the key the user saved",
      ).toEqual([["side_call", BLOCK_MODEL, true]]);
      expect(
        await toolResultTypes(page, sessionID),
        "the step should run once more after the card, not twice",
      ).toEqual(["setup_requirements", "block_output"]);
      await expect
        .poll(() => savedConversation(page, sessionID), {
          message: "the chat should save what was said, once each, in order",
        })
        .toEqual([
          ["user", `${FIRST_MESSAGE} ${run.marker}`],
          ["assistant", texts[0]],
          ["assistant", texts[1]],
          ["user", CONNECTED],
          ["assistant", texts[2]],
          ["assistant", texts[3]],
        ]);

      await page.reload();
      for (const text of [...streamed, CONNECTED]) {
        await expect(
          page.getByRole("log").getByText(text, { exact: true }),
          "the reloaded chat should show what was streamed, once",
        ).toHaveCount(1);
      }

      await openIntegrations(page);
      await expectListedInIntegrations(page, credential, SERVICE);

      const journal = await readJournal(run.nonce);
      expectEachTurnOnce(journal, CONNECT);
      await expectChargedWhatWasBilled(userID(journeyUser), [run.nonce]);
    } finally {
      await deleteCredential(page, credential);
    }
  });

  test.fail(
    "after a reload, the finished step's connect card stays gone and nothing offers to run the step again",
    {
      annotation: {
        type: "issue",
        description:
          "https://github.com/Significant-Gravitas/AutoGPT/issues/17253",
      },
    },
    async ({ page, journeyUser: _user }) => {
      const run = scriptedRun("connect-card-api-key");
      const finished = CONNECT.turns[3].text ?? "";

      const sessionID = await askForTheLine(page, run.marker, CONNECT);
      const credential = await connectKeyOnTheCard(page, {
        key: fakeAPIKey(),
        keyName: `Journey key ${run.nonce}`,
      });
      try {
        await expect(
          page.getByRole("log").getByText(finished, { exact: true }),
          "the chat should carry on by itself and finish",
        ).toHaveCount(1, { timeout: MODEL_TURN_TIMEOUT });
        await expectChatSettled(page, sessionID);

        // The card's row picks the saved key once the key list has loaded.
        const keysLoaded = page.waitForResponse(isCredentialsList);
        await page.reload();
        await keysLoaded;
        const reloaded = page.getByRole("log");
        await expect(reloaded.getByText(finished, { exact: true })).toHaveCount(
          1,
        );
        await expect(
          connectCards(reloaded),
          "the finished step's connect card should not come back after a reload",
        ).toHaveCount(0, { timeout: REDO_CHECK_TIMEOUT });
        await expect(
          reloaded.getByRole("button", { name: /^(Connect|Proceed)$/ }),
          "nothing should offer to run the finished step again",
        ).toHaveCount(0, { timeout: REDO_CHECK_TIMEOUT });
      } finally {
        await deleteCredential(page, credential);
      }
    },
  );

  test("cancelling the key on the card saves nothing and sends nothing: the chat stays where Otto said it needs the key, nothing is left running, and the next message is answered", async ({
    page,
    journeyUser,
  }) => {
    const run = scriptedRun("connect-card-cancelled");
    const texts = CANCEL.turns.map((turn) => turn.text ?? "");

    await recordChatStreams(page);
    const sessionID = await askForTheLine(page, run.marker, CANCEL);
    const transcript = page.getByRole("log");

    await connectButtons(transcript).click();
    const dialog = page.getByRole("dialog");
    await fillKeyForm(dialog, {
      key: fakeAPIKey(),
      keyName: `Journey key ${run.nonce}`,
    });
    await dialog.getByRole("button", { name: "Cancel" }).click();
    await expect(dialog, "Cancel should close the dialog").toBeHidden();

    await expect(
      connectCards(transcript),
      "the card should still ask for the account",
    ).toHaveCount(1);
    await expect(
      transcript.getByText("0 of 1 ready", { exact: true }),
      "nothing should be connected",
    ).toBeVisible();
    await expect(
      transcript.getByText(STILL_WAITING, { exact: true }),
      "the card should say the chat can't go on without the account",
    ).toBeVisible();
    await expect(
      connectButtons(transcript),
      "the card should still offer to connect",
    ).toBeVisible();
    expect(
      await savedKeys(page),
      "a cancelled key should not be saved",
    ).toEqual([]);
    await expectChatSettled(page, sessionID);

    const input = page.getByRole("textbox", { name: "Chat message input" });
    await input.fill(SKIP);
    await input.press("Enter");
    await expect(
      transcript.getByText(texts[2], { exact: true }),
      "the chat should answer the user's next message, once",
    ).toHaveCount(1, { timeout: MODEL_TURN_TIMEOUT });
    await expectChatSettled(page, sessionID);
    await expect
      .poll(() => savedConversation(page, sessionID), {
        message:
          "the chat should hold only what the user and Otto said: no follow-up from the card",
      })
      .toEqual([
        ["user", `${FIRST_MESSAGE} ${run.marker}`],
        ["assistant", texts[0]],
        ["assistant", texts[1]],
        ["user", SKIP],
        ["assistant", texts[2]],
      ]);
    expect(
      await chatStreamCount(page, "POST"),
      "only the user's own messages should be sent",
    ).toBe(2);
    expect(
      await toolResultTypes(page, sessionID),
      "the step should not run again",
    ).toEqual(["setup_requirements"]);

    const journal = await readJournal(run.nonce);
    expectEachTurnOnce(journal, CANCEL);
    await expectChargedWhatWasBilled(userID(journeyUser), [run.nonce]);
  });
});

/**
 * Asks Otto for the line. Its tool call needs an Anthropic key the user
 * doesn't have, so the turn ends at one connect card asking for it.
 */
async function askForTheLine(page: Page, marker: string, script: ModelScript) {
  await startChat(page, `${FIRST_MESSAGE} ${marker}`);
  const transcript = page.getByRole("log");
  await expect(
    transcript.getByText(script.turns[1].text ?? "", { exact: true }),
    "Otto should stop at the card and say what it needs",
  ).toBeVisible({ timeout: MODEL_TURN_TIMEOUT });
  await expect(
    connectCards(transcript),
    "the chat should show one connect card",
  ).toHaveCount(1);
  await expect(
    transcript.getByText("0 of 1 ready", { exact: true }),
    "the card should ask for one account, not yet connected",
  ).toBeVisible();
  await expect(
    transcript.getByText(SERVICE, { exact: true }).filter({ visible: true }),
    "the card should ask for Anthropic",
  ).toBeVisible();
  await expect(
    connectButtons(transcript),
    "the card should offer to connect it, once",
  ).toHaveCount(1);
  // The card asks for the account only, never the block's inputs (#14448):
  // a card with inputs gets a Proceed to send them.
  await expect(transcript.getByRole("button", { name: "Proceed" })).toHaveCount(
    0,
  );
  const sessionID = await currentSessionID(page);
  await expectChatSettled(page, sessionID);
  expect(await savedKeys(page), "the user has no Anthropic key yet").toEqual(
    [],
  );
  return sessionID;
}

function connectCards(scope: Locator) {
  return scope.getByRole("heading", { name: CARD_HEADING });
}

function connectButtons(scope: Locator) {
  return scope.getByRole("button", { name: "Connect", exact: true });
}

interface NewKey {
  key: string;
  keyName: string;
}

/** Adds the key through the card's Connect dialog. */
async function connectKeyOnTheCard(page: Page, newKey: NewKey) {
  await connectButtons(page.getByRole("log")).click();
  const dialog = page.getByRole("dialog");
  await fillKeyForm(dialog, newKey);
  const saved = page.waitForResponse(isKeySave);
  await dialog.getByRole("button", { name: "Continue" }).click();
  const response = await saved;
  expect(response.ok(), `saving the key: ${response.status()}`).toBe(true);
  expect(
    await response.text(),
    "saving a key should not answer with the key",
  ).not.toContain(newKey.key);
  await expect(dialog, "saving the key should close the dialog").toBeHidden();
  return (await response.json()) as Credential;
}

async function fillKeyForm(dialog: Locator, { key, keyName }: NewKey) {
  await expect(dialog.getByText(`Connect AutoGPT to ${SERVICE}`)).toBeVisible();
  await dialog.getByRole("button", { name: /API Key/ }).click();
  await dialog
    .getByRole("textbox", { name: "Name", exact: true })
    .fill(keyName);
  await dialog.getByLabel("API key", { exact: true }).fill(key);
}

function isKeySave(response: Response) {
  return (
    response.request().method() === "POST" &&
    new URL(response.url()).pathname.endsWith(
      `/api/integrations/${PROVIDER}/credentials`,
    )
  );
}

function isCredentialsList(response: Response) {
  return (
    response.request().method() === "GET" &&
    new URL(response.url()).pathname.endsWith("/api/integrations/credentials")
  );
}

async function savedKeys(page: Page) {
  const { credentials } = await listCredentials(page);
  return credentials
    .filter((credential) => credential.provider === PROVIDER)
    .map((credential) => [credential.title, credential.type]);
}

async function savedConversation(page: Page, sessionID: string) {
  return (await savedMessages(page, sessionID))
    .filter(
      (message) =>
        ["user", "assistant"].includes(message.role) && message.content,
    )
    .map((message) => [message.role, message.content]);
}

/** What each saved tool result was: the sign-in card, the block's output. */
async function toolResultTypes(page: Page, sessionID: string) {
  return (await savedMessages(page, sessionID))
    .filter((message) => message.role === "tool" && message.content)
    .map((message) => parseJSON(message.content as string)?.type);
}

function userID(user: { id?: string }) {
  if (!user.id) throw new Error("the journey user should have an id");
  return user.id;
}

function parseJSON(text: string) {
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return null;
  }
}
