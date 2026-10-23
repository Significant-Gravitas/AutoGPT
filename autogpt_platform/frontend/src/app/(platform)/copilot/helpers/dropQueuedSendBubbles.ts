import type { UIMessage } from "ai";

import { extractDbSequence } from "./convertChatSessionToUiMessages";

const PROMOTED_ID_PREFIX = "promoted-";

/**
 * Drop the bubbles of sends the server queued instead of storing.
 *
 * A send the server answers with "queued behind the running turn" keeps the
 * bubble `useChat` drew for it, yet no row backs it: the pending strip shows
 * the same message as a chip, and a reload shows only the chip. A local user
 * bubble after the last stored row whose text the queue holds is that send,
 * so the chip stands in for it. Each queued message accounts for one bubble,
 * the newest with its text.
 */
export function dropQueuedSendBubbles(
  messages: UIMessage[],
  queuedTexts: readonly string[],
): UIMessage[] {
  if (queuedTexts.length === 0) return messages;
  const lastStored = messages.findLastIndex(
    (message) => extractDbSequence(message) !== null,
  );
  const unclaimed = [...queuedTexts];
  const dropped = new Set<number>();
  // Newest first: the queued send is the latest bubble with its text, and an
  // earlier one with the same words was already answered.
  for (let index = messages.length - 1; index > lastStored; index--) {
    const message = messages[index];
    if (!isLocalUserBubble(message)) continue;
    const at = unclaimed.indexOf(textOf(message));
    if (at === -1) continue;
    unclaimed.splice(at, 1);
    dropped.add(index);
  }
  if (dropped.size === 0) return messages;
  return messages.filter((_, index) => !dropped.has(index));
}

function isLocalUserBubble(message: UIMessage) {
  return (
    message.role === "user" &&
    extractDbSequence(message) === null &&
    !message.id.startsWith(PROMOTED_ID_PREFIX)
  );
}

function textOf(message: UIMessage) {
  return message.parts
    .map((part) => (part.type === "text" ? part.text : ""))
    .join("")
    .trim();
}
