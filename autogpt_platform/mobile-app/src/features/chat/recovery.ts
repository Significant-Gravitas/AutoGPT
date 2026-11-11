import type { UIMessage } from "ai";

// Relative imports: jest has no `@/` alias, and nothing here may load the
// auth client, the AI SDK runtime or a native module.
import { SignedOutError } from "../../lib/auth/errors";
import { isStatusMessage } from "./chatErrors";

/** A network drop or a broken stream; not a dead session or a refusal. */
export function isRecoverableChatError(error: Error) {
  if (error instanceof SignedOutError) return false;
  return !isStatusMessage(error.message);
}

/**
 * True when the server took the turn the local chat last sent: it is still
 * running, or the server holds as many user messages as the chat, the last
 * one is the sent text, and a reply follows it. A repeated "ok" that never
 * left the phone, or a message stored without a reply, is not landed.
 */
export function turnLanded(
  rawMessages: unknown[],
  localMessages: UIMessage[],
  activeStream: object | null,
) {
  if (activeStream !== null) return true;
  const sent = lastUserText(localMessages);
  if (sent === null) return false;
  const roles = rawMessages.map(rowRole);
  const serverUsers = roles.filter((role) => role === "user").length;
  const localUsers = localMessages.filter((m) => m.role === "user").length;
  if (serverUsers < localUsers) return false;
  const lastUser = roles.lastIndexOf("user");
  const content = rowContent(rawMessages[lastUser]);
  if (content === null || !content.startsWith(sent)) return false;
  return roles
    .slice(lastUser + 1)
    .some((role) => role === "assistant" || role === "reasoning");
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

function rowRole(row: unknown) {
  if (typeof row !== "object" || row === null || !("role" in row)) return null;
  return typeof row.role === "string" ? row.role : null;
}

function rowContent(row: unknown) {
  if (typeof row !== "object" || row === null || !("content" in row)) {
    return null;
  }
  return typeof row.content === "string" ? row.content : null;
}
