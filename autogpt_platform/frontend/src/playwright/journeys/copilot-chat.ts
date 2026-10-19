import {
  expect,
  type Page,
  type Request as PlaywrightRequest,
} from "@playwright/test";

/** Opens AutoPilot and sends the first message of a new chat. */
export async function startChat(page: Page, message: string) {
  await page.goto("/home");
  const input = page.getByRole("textbox", { name: "Chat message input" });
  await expect(input).toBeVisible();
  await input.fill(message);
  await input.press("Enter");
}

interface RecordedStream {
  method: string;
  text: string;
  done: boolean;
}

/**
 * Records the body of every chat stream the page reads, as the browser
 * received it, and lets a journey cut the open ones the way a dropped
 * connection would (cutChatStreams). Chromium neither keeps a streamed
 * response's body for Playwright nor fails an open stream when the context
 * goes offline, so the page's fetch hands the app a stream the journey
 * controls. Call before the page loads.
 */
export async function recordChatStreams(page: Page) {
  await page.context().addInitScript(() => {
    const streams: RecordedStream[] = [];
    const openStreams = new Set<() => void>();
    Object.defineProperty(window, "__journeyChatStreams", { value: streams });
    Object.defineProperty(window, "__journeyCutChatStreams", {
      value: () => {
        const cuts = [...openStreams];
        cuts.forEach((cut) => cut());
        return cuts.length;
      },
    });
    const pageFetch = window.fetch.bind(window);
    window.fetch = async (input, init) => {
      const response = await pageFetch(input, init);
      const url = input instanceof Request ? input.url : String(input);
      const method = (
        init?.method ?? (input instanceof Request ? input.method : "GET")
      ).toUpperCase();
      const { pathname } = new URL(url, window.location.href);
      if (!/\/api\/chat\/sessions\/[^/]+\/stream$/.test(pathname)) {
        return response;
      }
      if (!response.body) return response;
      const record: RecordedStream = { method, text: "", done: false };
      streams.push(record);
      const upstream = response.body.getReader();
      const decoder = new TextDecoder();
      let cutOff = false;
      let pageSide: ReadableStreamDefaultController<Uint8Array> | null = null;
      function end() {
        record.done = true;
        openStreams.delete(drop);
      }
      function drop() {
        cutOff = true;
        end();
        void upstream.cancel();
        // What the app's reader sees when the connection drops.
        pageSide?.error(new TypeError("network error"));
      }
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          pageSide = controller;
          openStreams.add(drop);
        },
        async pull(controller) {
          try {
            const { done, value } = await upstream.read();
            if (cutOff) return;
            if (done) {
              end();
              controller.close();
              return;
            }
            record.text += decoder.decode(value, { stream: true });
            controller.enqueue(value);
          } catch (error) {
            if (cutOff) return;
            end();
            controller.error(error);
          }
        },
        cancel(reason) {
          end();
          return upstream.cancel(reason);
        },
      });
      return new Response(body, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    };
  });
}

/** Fails every open chat stream as a dropped connection would. */
export async function cutChatStreams(page: Page) {
  const cut = await page.evaluate(() =>
    (
      window as unknown as { __journeyCutChatStreams: () => number }
    ).__journeyCutChatStreams(),
  );
  expect(cut, "a chat stream should be open to cut").toBeGreaterThan(0);
}

/** The body of the first chat stream sent with `method`, once it has ended. */
export async function finishedChatStream(page: Page, method: "GET" | "POST") {
  const handle = await page.waitForFunction(
    (wanted) => {
      const streams = (
        window as unknown as { __journeyChatStreams?: RecordedStream[] }
      ).__journeyChatStreams;
      const stream = streams?.find((s) => s.method === wanted && s.done);
      return stream ? stream.text : null;
    },
    method,
    { timeout: 60_000 },
  );
  return (await handle.jsonValue()) as string;
}

/** The start of `text`, to see an answer begin streaming. */
export function firstWords(text: string, count = 5) {
  return text.split(" ").slice(0, count).join(" ");
}

/** The request that sends a message and streams the turn back (POST), or
 * reconnects to a running turn (GET). It goes to the backend directly. */
export function isChatStream(method: "GET" | "POST") {
  return (request: PlaywrightRequest) =>
    request.method() === method &&
    /\/api\/chat\/sessions\/[^/]+\/stream$/.test(
      new URL(request.url()).pathname,
    );
}

export async function currentSessionID(page: Page) {
  await expect
    .poll(() => new URL(page.url()).searchParams.get("sessionId"), {
      message: "the chat should get a session",
    })
    .toBeTruthy();
  return new URL(page.url()).searchParams.get("sessionId") as string;
}

export interface SavedMessage {
  role: string;
  content: string | null;
}

export async function savedMessages(page: Page, sessionID: string) {
  const response = await page.request.get(
    `/api/proxy/api/chat/sessions/${sessionID}`,
  );
  expect(response.ok(), `GET chat session: ${response.status()}`).toBe(true);
  const body = (await response.json()) as { messages: SavedMessage[] };
  return body.messages;
}

/** The assistant's text, one entry per saved message that has any. */
export async function savedAssistantTexts(page: Page, sessionID: string) {
  return (await savedMessages(page, sessionID))
    .filter((message) => message.role === "assistant" && message.content)
    .map((message) => message.content);
}
