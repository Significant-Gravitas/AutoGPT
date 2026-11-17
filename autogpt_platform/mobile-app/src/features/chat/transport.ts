import { DefaultChatTransport, type ChatTransport, type UIMessage } from "ai";
import * as Crypto from "expo-crypto";
import { fetch as expoFetch } from "expo/fetch";

import { apiUrl, authHeaders } from "@/lib/api/client";
import { SessionChangedError, SignedOutError } from "@/lib/auth/errors";
import { assertSameSession, clearAccessToken, sessionMark } from "@/lib/auth/token";
import { tapReply } from "@/lib/haptics";

import { messageFiles } from "./attachmentHelpers";
import { OFFLINE_MESSAGE, statusMessage } from "./chatErrors";
import { paceTextStream } from "./paceStream";

/**
 * Points the AI SDK at the backend's stream route, like the web app does, and
 * lets the reply's text out at a steady pace.
 */
export function createChatTransport(sessionID: string): ChatTransport<UIMessage> {
  const inner = createStreamTransport(sessionID);
  return {
    async sendMessages(options) {
      const stream = await inner.sendMessages(options);
      return paceTextStream(stream, { onFirstText: tapReply });
    },
    async reconnectToStream(options) {
      const stream = await inner.reconnectToStream(options);
      return stream ? paceTextStream(stream) : null;
    },
  };
}

function createStreamTransport(sessionID: string) {
  const api = apiUrl(
    `/api/chat/sessions/${encodeURIComponent(sessionID)}/stream`,
  );

  return new DefaultChatTransport<UIMessage>({
    api,
    fetch: fetchStream,
    prepareSendMessagesRequest: async ({ messages }) => ({
      body: streamRequestBody(messages[messages.length - 1]),
      headers: await streamHeaders(),
    }),
    // Always from the turn's first chunk: the parser needs each part's start.
    prepareReconnectToStreamRequest: async () => ({
      api,
      headers: await streamHeaders(),
    }),
  });
}

function streamRequestBody(last: UIMessage | undefined) {
  const text = (last?.parts ?? [])
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("");
  const { fileIDs } = messageFiles(last?.metadata);
  return {
    message: text,
    is_user_message: true,
    // One id per send, so a retransmit never starts a second turn.
    message_id: Crypto.randomUUID(),
    ...(fileIDs.length ? { file_ids: fileIDs } : {}),
  };
}

// A dead session keeps its own sentence; any other failure to mint a token
// means the server could not be reached.
async function streamHeaders() {
  try {
    return await authHeaders();
  } catch (error) {
    if (error instanceof SignedOutError || error instanceof SessionChangedError) throw error;
    throw new Error(OFFLINE_MESSAGE);
  }
}

// The SDK reports a refused request as a bare Error holding the body, so the
// reason is put into words here.
const fetchStream: typeof globalThis.fetch = async (input, init) => {
  const mark = sessionMark();
  let response = await send(input, init);
  // A refused token is dropped and the request sent once more with a fresh
  // one; the body keeps its message_id, so this cannot start a second turn.
  if (response.status === 401) {
    // Never post the message for an account that signed in after it was sent.
    assertSameSession(mark);
    clearAccessToken();
    response = await send(input, withHeaders(init, await streamHeaders()));
    if (response.status === 401) {
      // Signs out only if the session that sent this is still the current one.
      assertSameSession(mark);
      throw new SignedOutError();
    }
  }
  if (response.ok) return response;
  throw new Error(statusMessage(response.status));
};

async function send(input: RequestInfo | URL, init: RequestInit | undefined) {
  try {
    return await (expoFetch as unknown as typeof globalThis.fetch)(input, init);
  } catch (error) {
    // `chat.stop()` aborts the request; the SDK must still see an abort.
    if (isAbort(error, init)) throw error;
    throw new Error(OFFLINE_MESSAGE);
  }
}

// The SDK lowercases header names, so a plain spread would send the old
// `authorization` beside the new `Authorization`.
function withHeaders(
  init: RequestInit | undefined,
  fresh: Record<string, string>,
): RequestInit {
  const headers = new Headers(init?.headers);
  for (const [name, value] of Object.entries(fresh)) headers.set(name, value);
  return { ...init, headers };
}

function isAbort(error: unknown, init: RequestInit | undefined) {
  if (init?.signal?.aborted) return true;
  return error instanceof Error && error.name === "AbortError";
}
