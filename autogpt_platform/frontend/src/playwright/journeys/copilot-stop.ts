import { expect, type Page, type Response } from "@playwright/test";
import {
  firstWords,
  isChatStream,
  type RecordedStream,
  savedMessages,
} from "./copilot-chat";

/** Starts every error the backend writes into a reply, a stopped turn's
 *  "Operation cancelled" among them. */
const ERROR_MARKER = "[__COPILOT_ERROR_";

/** Clicks Stop on the turn being written; returns when it was clicked. */
export async function clickStop(page: Page) {
  await composer(page)
    .getByRole("button", { name: "Stop", exact: true })
    .click();
  return Date.now();
}

/** The chat's composer: its message box and the buttons beside it. */
function composer(page: Page) {
  return page.locator("form").filter({ has: messageBox(page) });
}

function messageBox(page: Page) {
  return page.getByRole("textbox", { name: "Chat message input" });
}

/**
 * The server's answer to the cancel a Stop sends, and when it came. The
 * cancel route waits up to 5 s for the executor to confirm the stop; a
 * `reason` of "cancel_published_not_confirmed" says it gave up waiting and
 * completed the turn itself. Call before the Stop.
 */
export function watchCancel(page: Page, timeout: number) {
  const answered = page
    .waitForResponse(isCancel, { timeout })
    .then(async (response) => {
      const at = Date.now();
      const body = (await response.json().catch(() => ({}))) as {
        reason?: string | null;
      };
      return { at, status: response.status(), reason: body.reason ?? null };
    });
  void answered.catch(() => undefined);
  return answered;
}

function isCancel(response: Response) {
  return (
    response.request().method() === "POST" &&
    /\/api\/chat\/sessions\/[^/]+\/cancel$/.test(
      new URL(response.url()).pathname,
    )
  );
}

/**
 * Types `text` and sends it as a new message: the composer must offer Send,
 * not Stop or Queue. Returns when Send was ready.
 */
export async function sendAsNewMessage(page: Page, text: string) {
  const input = messageBox(page);
  await input.fill(text);
  await expect(
    composer(page).getByRole("button", { name: "Submit", exact: true }),
    "the composer should take a new message",
  ).toBeEnabled();
  const ready = Date.now();
  const sent = page.waitForRequest(isChatStream("POST"));
  void sent.catch(() => undefined);
  await input.press("Enter");
  await sent;
  return ready;
}

/** How many of `answer`'s words, from its start, `text` holds. */
export function answerWordsIn(text: string, answer: string) {
  const words = answer.split(" ").length;
  let held = 0;
  while (held < words && text.includes(firstWords(answer, held + 1))) {
    held += 1;
  }
  return held;
}

/**
 * The saved chat as [role, text] pairs, a turn's assistant messages joined
 * into one and cut at the first error marker the backend wrote into them.
 */
export async function savedConversation(page: Page, sessionID: string) {
  const turns: [string, string][] = [];
  for (const message of await savedMessages(page, sessionID)) {
    if (message.role !== "user" && message.role !== "assistant") continue;
    const last = turns[turns.length - 1];
    if (message.role === "assistant" && last?.[0] === "assistant") {
      last[1] += message.content ?? "";
    } else {
      turns.push([message.role, message.content ?? ""]);
    }
  }
  return turns.map(
    ([role, text]) => [role, text.split(ERROR_MARKER)[0].trim()] as const,
  );
}

/** Error cards in the chat, such as a turn that failed to start. */
export function errorCards(page: Page) {
  return page.getByRole("log").locator('[data-slot="error-card-content"]');
}

/** Checks the chat shows each of `texts` once, in this order. */
export async function expectOnceEachInOrder(
  page: Page,
  texts: string[],
  message: string,
) {
  const shown = await page.getByRole("log").innerText();
  expect(
    texts.map((text) => shown.split(text).length - 1),
    `${message}: each once`,
  ).toEqual(texts.map(() => 1));
  const at = texts.map((text) => shown.indexOf(text));
  expect(
    [...at].sort((a, b) => a - b),
    `${message}: in order`,
  ).toEqual(at);
}

/**
 * What became of the page's `index`th send (0 = its first message), as the
 * chat stream recorder (recordChatStreams) saw it: queued behind a running
 * turn, or a stream of its own.
 */
export async function sendOutcome(page: Page, index: number) {
  return page.evaluate((index) => {
    const streams =
      (window as unknown as { __journeyChatStreams?: RecordedStream[] })
        .__journeyChatStreams ?? [];
    const sent = streams.filter((stream) => stream.method === "POST")[index];
    if (!sent) return "no stream";
    if (sent.text.includes("data-follow-up-queued")) {
      return "queued behind the running turn";
    }
    return `its own stream (${sent.done ? "ended" : "open"}, ${sent.text.length} bytes)`;
  }, index);
}
