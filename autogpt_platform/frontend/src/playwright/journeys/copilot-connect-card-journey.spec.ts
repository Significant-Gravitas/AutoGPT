import { randomBytes } from "crypto";
import type { Locator, Page } from "@playwright/test";
import {
  chatStreamCount,
  currentSessionID,
  expectChatSettled,
  finishedChatStreams,
  recordChatStreams,
  savedMessages,
  startChat,
} from "./copilot-chat";
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
// LLM keys, so the block needs the user's own Anthropic key and answers with
// its sign-in card. The journeys stack points Anthropic clients at the
// scripted model, which journals which key each request presented.
const CONNECT: ModelScript = connectScript;
const CANCEL: ModelScript = cancelScript;
const FIRST_MESSAGE = "Write me one line about the sea with Claude.";
const CARD_HEADING = "Plug in what this needs";
// What the card sends once the key is saved (SetupRequirementsCard helpers).
const CONNECTED = CONNECT.turns[2].expect_user_text ?? "";
const SKIP = CANCEL.turns[2].expect_user_text ?? "";
const BLOCK_MODEL = "claude-haiku-4-5-20251001";

test.describe("copilot connect card journey", () => {
  test.beforeAll(requireScriptedModel);

  test("Otto asks for an API key, the user connects it on the card, and the chat carries on with it", async ({
    page,
    journeyUser,
  }) => {
    const run = scriptedRun("connect-card-api-key");
    const key = newAPIKey();
    const keyName = `Journey key ${run.nonce}`;
    const texts = CONNECT.turns.map((turn) => turn.text ?? "");

    await recordChatStreams(page);
    const sessionID = await askForTheLine(page, run.marker);
    await connectKeyOnTheCard(page, { key, keyName });

    const transcript = page.getByRole("log");
    await expect(
      transcript.getByText(texts[3], { exact: true }),
      "the chat should carry on by itself and finish, once",
    ).toHaveCount(1);
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
    const keyCalls = await readJournalForKey(key);
    expect(
      keyCalls.map((call) => [call.kind, call.model, call.completed]),
      "the block should call the provider once, with the key the user saved",
    ).toEqual([["side_call", BLOCK_MODEL, true]]);
    expect(
      await toolResultTypes(page, sessionID),
      "the block should run once more after the card, not twice",
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

    await page.goto("/settings/integrations");
    await expect(
      page.getByRole("checkbox", { name: `Select ${keyName}` }),
      "Settings > Integrations should list the key, once",
    ).toHaveCount(1);

    const journal = await readJournal(run.nonce);
    expectEachTurnOnce(journal, CONNECT);
    await expectChargedWhatWasBilled(userID(journeyUser), journal);
  });

  test("after a reload, a step the user already connected for offers nothing to redo", async ({
    page,
  }) => {
    const run = scriptedRun("connect-card-api-key");
    const finished = CONNECT.turns[3].text ?? "";

    const sessionID = await askForTheLine(page, run.marker);
    await connectKeyOnTheCard(page, {
      key: newAPIKey(),
      keyName: `Journey key ${run.nonce}`,
    });
    await expect(
      page.getByRole("log").getByText(finished, { exact: true }),
    ).toBeVisible();
    await expectChatSettled(page, sessionID);

    await page.reload();
    const transcript = page.getByRole("log");
    await expect(transcript.getByText(finished, { exact: true })).toBeVisible();
    await expect(
      transcript.getByText(CARD_HEADING, { exact: true }),
      "the connected step's card should not come back after a reload",
    ).toHaveCount(0);
    await expect(
      transcript.getByRole("button", { name: /^(Connect|Proceed)$/ }),
      "nothing should offer to run the finished step again",
    ).toHaveCount(0);
  });

  test("cancelling the key leaves nothing held: no key is saved, nothing is sent, and the chat takes the user's next message", async ({
    page,
    journeyUser,
  }) => {
    const run = scriptedRun("connect-card-cancelled");
    const key = newAPIKey();
    const texts = CANCEL.turns.map((turn) => turn.text ?? "");

    await recordChatStreams(page);
    const sessionID = await askForTheLine(page, run.marker);
    const transcript = page.getByRole("log");

    await transcript
      .getByRole("button", { name: "Connect", exact: true })
      .click();
    const dialog = page.getByRole("dialog");
    await fillKeyForm(dialog, { key, keyName: `Journey key ${run.nonce}` });
    await dialog.getByRole("button", { name: "Cancel" }).click();
    await expect(dialog, "Cancel should close the dialog").toBeHidden();

    await expect(
      transcript.getByRole("button", { name: "Connect", exact: true }),
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
    ).toHaveCount(1);
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
      await readJournalForKey(key),
      "nothing should run with a cancelled key",
    ).toEqual([]);
    expect(
      await toolResultTypes(page, sessionID),
      "the block should not run again",
    ).toEqual(["setup_requirements"]);

    const journal = await readJournal(run.nonce);
    expectEachTurnOnce(journal, CANCEL);
    await expectChargedWhatWasBilled(userID(journeyUser), journal);
  });
});

/**
 * Asks Otto for the line. Its tool call yields one connect card for
 * Anthropic, asking for the account only, and the turn ends there.
 */
async function askForTheLine(page: Page, marker: string) {
  await startChat(page, `${FIRST_MESSAGE} ${marker}`);
  const transcript = page.getByRole("log");
  await expect(
    transcript.getByText(CONNECT.turns[1].text ?? "", { exact: true }),
    "Otto should stop at the card and say what it needs",
  ).toBeVisible();
  await expect(
    transcript.getByText(CARD_HEADING, { exact: true }),
    "the chat should show one connect card",
  ).toHaveCount(1);
  await expect(
    transcript.getByRole("button", { name: "Connect", exact: true }),
    "the card should offer to connect Anthropic",
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

interface NewKey {
  key: string;
  keyName: string;
}

/** Adds the key through the card's Connect dialog. */
async function connectKeyOnTheCard(page: Page, newKey: NewKey) {
  const transcript = page.getByRole("log");
  await transcript
    .getByRole("button", { name: "Connect", exact: true })
    .click();
  const dialog = page.getByRole("dialog");
  await fillKeyForm(dialog, newKey);
  await dialog.getByRole("button", { name: "Continue" }).click();
  await expect(dialog, "saving the key should close the dialog").toBeHidden();
}

async function fillKeyForm(dialog: Locator, { key, keyName }: NewKey) {
  await expect(dialog.getByText("Connect AutoGPT to Anthropic")).toBeVisible();
  await dialog.getByRole("button", { name: /API Key/ }).click();
  await dialog.getByLabel("Name", { exact: true }).fill(keyName);
  await dialog.getByPlaceholder("sk-...").fill(key);
}

/** A fresh key per run: the scripted model journals which key each call
 * used, so a run finds its own calls by it. */
function newAPIKey() {
  return `sk-ant-journey-${randomBytes(16).toString("hex")}`;
}

async function savedKeys(page: Page) {
  const response = await page.request.get(
    "/api/proxy/api/integrations/anthropic/credentials",
  );
  expect(response.ok(), `GET anthropic credentials: ${response.status()}`).toBe(
    true,
  );
  const body = (await response.json()) as { title: string; type: string }[];
  return body.map((credential) => [credential.title, credential.type]);
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
