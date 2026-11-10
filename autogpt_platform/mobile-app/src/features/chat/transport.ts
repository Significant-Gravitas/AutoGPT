import { DefaultChatTransport, type UIMessage } from "ai";
import * as Crypto from "expo-crypto";
import { fetch as expoFetch } from "expo/fetch";

import { apiUrl, authHeaders } from "@/lib/api/client";

const STATUS_MESSAGES: Record<number, string> = {
  401: "You are signed out. Sign in again to continue.",
  402: "Chat needs an active plan. You can manage your plan on the web.",
  404: "This chat no longer exists.",
  429: "You have reached your usage limit. Try again later.",
};

/** Points the AI SDK at the backend's stream route, like the web app does. */
export function createChatTransport(sessionID: string) {
  const api = apiUrl(`/api/chat/sessions/${sessionID}/stream`);

  return new DefaultChatTransport<UIMessage>({
    api,
    fetch: fetchStream,
    prepareSendMessagesRequest: async ({ messages }) => ({
      body: streamRequestBody(messages[messages.length - 1]),
      headers: await authHeaders(),
    }),
    // Always from the turn's first chunk: the parser needs each part's start.
    prepareReconnectToStreamRequest: async () => ({
      api,
      headers: await authHeaders(),
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

// The SDK reports a refused request as a bare Error holding the body, so the
// reason is put into words here.
const fetchStream: typeof globalThis.fetch = async (input, init) => {
  const response = await (expoFetch as unknown as typeof globalThis.fetch)(
    input,
    init,
  );
  if (response.ok) return response;
  throw new Error(
    STATUS_MESSAGES[response.status] ??
      "Otto could not reply just now. Try again in a moment.",
  );
};
