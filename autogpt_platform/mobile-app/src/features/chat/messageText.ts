import type { UIMessage } from "ai";

/** The words of a message, as a person would copy them. */
export function messageText(message: UIMessage) {
  return message.parts
    .flatMap((part) => (part.type === "text" && part.text.trim() ? [part.text.trim()] : []))
    .join("\n\n");
}

/** A message this screen sent or streamed, rather than one loaded from the server. */
export function isLocalMessage(message: UIMessage, sessionID: string) {
  return !message.id.startsWith(`${sessionID}-`);
}
