import { DefaultChatTransport, type UIMessage } from "ai";

import { normalizeSameOriginApiBaseURL, type AccessTokenProvider } from "./api";

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
  const baseURL = normalizeSameOriginApiBaseURL(apiBaseURL);
  const api = `${baseURL}/api/embed/v1/sessions/${sessionID}/stream`;
  return new DefaultChatTransport<UIMessage>({
    api,
    prepareSendMessagesRequest: async ({ messages }) => {
      const token = await getAccessToken();
      const lastMessage = messages[messages.length - 1];
      const messageID = createMessageID();
      return {
        body: {
          message: textFromMessage(lastMessage),
          ...(messageID ? { message_id: messageID } : {}),
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

export function createMessageID(): string | undefined {
  const webCrypto = globalThis.crypto;
  if (!webCrypto) return undefined;
  if (typeof webCrypto.randomUUID === "function") {
    return webCrypto.randomUUID();
  }
  if (typeof webCrypto.getRandomValues !== "function") return undefined;

  const bytes = webCrypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0"));
  return `${hex.slice(0, 4).join("")}-${hex.slice(4, 6).join("")}-${hex.slice(6, 8).join("")}-${hex.slice(8, 10).join("")}-${hex.slice(10).join("")}`;
}
