import { DefaultChatTransport, type UIMessage } from "ai";

import type { AccessTokenProvider } from "./api";

interface CreateEmbeddedTransportArgs {
  apiBaseURL: string;
  sessionID: string;
  getAccessToken: AccessTokenProvider;
}

export function createEmbeddedTransport({
  apiBaseURL,
  sessionID,
  getAccessToken,
}: CreateEmbeddedTransportArgs) {
  const api = `${apiBaseURL.replace(/\/+$/, "")}/api/embed/v1/sessions/${sessionID}/stream`;
  return new DefaultChatTransport<UIMessage>({
    api,
    prepareSendMessagesRequest: async ({ messages }) => {
      const token = await getAccessToken();
      const lastMessage = messages[messages.length - 1];
      return {
        body: {
          message: textFromMessage(lastMessage),
          message_id: crypto.randomUUID(),
        },
        headers: { Authorization: `Bearer ${token}` },
      };
    },
  });
}

export function textFromMessage(message: UIMessage | undefined): string {
  if (!message) return "";
  return message.parts
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("");
}
