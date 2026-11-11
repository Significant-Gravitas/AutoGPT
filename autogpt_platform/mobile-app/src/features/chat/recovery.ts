import type { UIMessage } from "ai";

import { SignedOutError } from "@/lib/auth/token";

import { isStatusMessage } from "./transport";

/** A network drop or a broken stream; not a dead session or a refusal. */
export function isRecoverableChatError(error: Error) {
  if (error instanceof SignedOutError) return false;
  return !isStatusMessage(error.message);
}

/** The backend may append an attached-files block, so only the start counts. */
export function serverHasUserMessage(rawMessages: unknown[], text: string) {
  for (let i = rawMessages.length - 1; i >= 0; i--) {
    const row = rawMessages[i];
    if (typeof row !== "object" || row === null) continue;
    if (!("role" in row) || row.role !== "user") continue;
    return (
      "content" in row &&
      typeof row.content === "string" &&
      row.content.startsWith(text)
    );
  }
  return false;
}

export function lastMessageIsUser(messages: UIMessage[]) {
  return messages[messages.length - 1]?.role === "user";
}

export function lastUserText(messages: UIMessage[]) {
  const last = messages.findLast((message) => message.role === "user");
  if (!last) return null;
  return last.parts
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("");
}
