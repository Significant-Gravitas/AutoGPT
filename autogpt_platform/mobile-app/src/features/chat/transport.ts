import { DefaultChatTransport, type UIMessage } from "ai";
import * as Crypto from "expo-crypto";
import { fetch as expoFetch } from "expo/fetch";

import { apiUrl, authHeaders } from "@/lib/api/client";
import { clearAccessToken, SignedOutError } from "@/lib/auth/token";

const OFFLINE_MESSAGE =
  "Can't reach AutoGPT. Check your connection and try again.";

const FALLBACK_MESSAGE =
  "Otto could not reply just now. Try again in a moment.";

export const PLAN_REQUIRED_MESSAGE =
  "Chat needs an active plan. You can manage your plan on the web.";

const STATUS_MESSAGES: Record<number, string> = {
  402: PLAN_REQUIRED_MESSAGE,
  404: "This chat no longer exists.",
  429: "You have reached your usage limit. Try again later.",
};

const KNOWN_MESSAGES = new Set([
  ...Object.values(STATUS_MESSAGES),
  OFFLINE_MESSAGE,
  FALLBACK_MESSAGE,
  new SignedOutError().message,
]);

/** The sentence to show for a chat error: ours as is, anything else generic. */
export function chatErrorMessage(error: Error | undefined) {
  if (!error) return null;
  return KNOWN_MESSAGES.has(error.message) ? error.message : FALLBACK_MESSAGE;
}

/** Points the AI SDK at the backend's stream route, like the web app does. */
export function createChatTransport(sessionID: string) {
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
  return {
    message: text,
    is_user_message: true,
    // One id per send, so a retransmit never starts a second turn.
    message_id: Crypto.randomUUID(),
  };
}

// A dead session keeps its own sentence; any other failure to mint a token
// means the server could not be reached.
async function streamHeaders() {
  try {
    return await authHeaders();
  } catch (error) {
    if (error instanceof SignedOutError) throw error;
    throw new Error(OFFLINE_MESSAGE);
  }
}

// The SDK reports a refused request as a bare Error holding the body, so the
// reason is put into words here.
const fetchStream: typeof globalThis.fetch = async (input, init) => {
  let response = await send(input, init);
  // A refused token is dropped and the request sent once more with a fresh
  // one; the body keeps its message_id, so this cannot start a second turn.
  if (response.status === 401) {
    clearAccessToken();
    response = await send(input, withHeaders(init, await streamHeaders()));
    if (response.status === 401) throw new SignedOutError();
  }
  if (response.ok) return response;
  throw new Error(STATUS_MESSAGES[response.status] ?? FALLBACK_MESSAGE);
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
