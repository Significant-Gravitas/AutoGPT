import type { Locator, Page, Response } from "@playwright/test";
import {
  currentSessionID,
  expectChatSettled,
  savedMessages,
  startChat,
} from "./copilot-chat";
import { type Credential, listCredentials } from "./credentials";
import { expect } from "./fixtures";
import type { ModelScript } from "./scripted-model";

// AutoPilot runs the AI Text Generator with Claude. The stack has no platform
// LLM keys, so the block needs the user's own Anthropic key and answers with a
// sign-in card, which the chat shows as one connect card under the step.
export const PROVIDER = "anthropic";
export const SERVICE = "Anthropic";
export const FIRST_MESSAGE = "Write me one line about the sea with Claude.";
// A model turn starts a Claude Agent SDK session, which takes a while on CI.
export const MODEL_TURN_TIMEOUT = 60_000;
const CARD_HEADING = /^(Connect your apps|Your apps are connected)$/;

export interface NewKey {
  key: string;
  keyName: string;
}

interface KeySave {
  response: Response;
  dialog: Locator;
  /** The saved key's record, or null when the save failed. */
  credential: Credential | null;
}

/**
 * Asks Otto for the line. Its tool call needs an Anthropic key the user
 * doesn't have, so the turn ends at one connect card asking for it.
 */
export async function askForTheLine(
  page: Page,
  marker: string,
  script: ModelScript,
) {
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

export function connectCards(scope: Locator) {
  return scope.getByRole("heading", { name: CARD_HEADING });
}

export function connectButtons(scope: Locator) {
  return scope.getByRole("button", { name: "Connect", exact: true });
}

/**
 * Adds the key through the card's Connect dialog. Resolves with what the save
 * answered before checking it (`expectKeySaved`), so a caller can delete the
 * key in its cleanup even when a check fails.
 */
export async function connectKeyOnTheCard(
  page: Page,
  newKey: NewKey,
): Promise<KeySave> {
  await connectButtons(page.getByRole("log")).click();
  const dialog = page.getByRole("dialog");
  await fillKeyForm(dialog, newKey);
  const saved = page.waitForResponse(isKeySave);
  await dialog.getByRole("button", { name: "Continue" }).click();
  const response = await saved;
  const credential = response.ok()
    ? ((await response.json()) as Credential)
    : null;
  return { response, dialog, credential };
}

/** The save answered with the key's record, not the key, and closed. */
export async function expectKeySaved(save: KeySave, newKey: NewKey) {
  expect(save.response.ok(), `saving the key: ${save.response.status()}`).toBe(
    true,
  );
  expect(
    await save.response.text(),
    "saving a key should not answer with the key",
  ).not.toContain(newKey.key);
  await expect(
    save.dialog,
    "saving the key should close the dialog",
  ).toBeHidden();
  if (!save.credential) throw new Error("the save should return the key");
  return save.credential;
}

export async function fillKeyForm(dialog: Locator, { key, keyName }: NewKey) {
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

export function isCredentialsList(response: Response) {
  return (
    response.request().method() === "GET" &&
    new URL(response.url()).pathname.endsWith("/api/integrations/credentials")
  );
}

export async function savedKeys(page: Page) {
  const { credentials } = await listCredentials(page);
  return credentials
    .filter((credential) => credential.provider === PROVIDER)
    .map((credential) => [credential.title, credential.type]);
}

export async function savedConversation(page: Page, sessionID: string) {
  return (await savedMessages(page, sessionID))
    .filter(
      (message) =>
        ["user", "assistant"].includes(message.role) && message.content,
    )
    .map((message) => [message.role, message.content]);
}

/** What each saved tool result was: the sign-in card, the block's output. */
export async function toolResultTypes(page: Page, sessionID: string) {
  return (await savedMessages(page, sessionID))
    .filter((message) => message.role === "tool" && message.content)
    .map((message) => parseJSON(message.content as string)?.type);
}

export function userID(user: { id?: string }) {
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
