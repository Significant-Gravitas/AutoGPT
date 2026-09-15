import type { UIDataTypes, UIMessage, UITools } from "ai";

import { isBookkeepingPart } from "../../messageParts";
import { getTurnMessages, type MessagePart } from "./helpers";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

export const PENDING_DRAINED_PART_TYPE = "data-pending-drained";

/** Marks a row that is only part of its source message. The final segment
 *  keeps the untouched message id, so everything keyed on it (turn stats,
 *  retry, minimap anchors, the thinking indicator) still finds its row. */
const SEGMENT_ID_MARKER = "#seg";

export function isMidTurnSegmentRow(message: ChatMessage): boolean {
  return message.id.includes(SEGMENT_ID_MARKER);
}

interface DrainedMessage {
  /** Backend-assigned id of the pending message; null on a malformed hint. */
  id: string | null;
  content: string;
}

/**
 * Render the mid-turn follow-up in chronological order: the tool chain that
 * ran before the drain, then the user's message, then the work that came
 * after it — the shape every chat app has.
 *
 * The array itself cannot be split that way. AI SDK's `useChat` accumulates
 * a whole turn into ONE assistant `UIMessage` and, on every delta, replaces
 * `messages[len - 1]` only while `lastMessage.id === state.message.id`;
 * otherwise it pushes a fresh copy of the whole turn. So a user bubble
 * inserted between the turn's parts either stops the stream from landing or
 * duplicates the entire chain. Splitting at render time instead leaves the
 * array untouched and keeps the transcript honest.
 *
 * The split point is the backend's `data-pending-drained` hint, which
 * carries the drained text (`data.messages`). A hint without it comes from
 * an older backend: no split, and `useCopilotPendingChips` keeps promoting
 * a bubble at the tail as before.
 */
export function splitMessagesAtDrainHints(
  messages: ChatMessage[],
): ChatMessage[] {
  if (!messagesCarryDrainedText(messages)) return messages;

  const rows: ChatMessage[] = [];
  for (const message of messages) {
    if (!hasSplittableDrainHint(message)) {
      rows.push(message);
      continue;
    }
    rows.push(...splitAssistantMessage(message));
  }
  return rows;
}

/**
 * Does the live stream carry the drained text itself? When it does, the
 * follow-up bubble comes from the split above and `useCopilotPendingChips`
 * must not promote a second one at the tail.
 */
export function messagesCarryDrainedText(messages: ChatMessage[]): boolean {
  return messages.some(hasSplittableDrainHint);
}

/**
 * The follow-up texts the split above actually draws, in transcript order.
 *
 * `useCopilotPendingChips` promotes a drained chip into a bubble only when
 * the transcript is not already showing it. Asking that per text rather than
 * "does this transcript contain any text-bearing hint" matters when the two
 * kinds are mixed — an older turn's text-bearing hint must not suppress the
 * bubble for a later count-only drain, whose entries are dropped from the
 * queue either way and would otherwise vanish until hydration.
 */
export function drainedTextsInMessages(messages: ChatMessage[]): string[] {
  const texts: string[] = [];
  for (const message of messages) {
    if (message.role !== "assistant") continue;
    let hasVisibleAbove = false;
    for (const part of message.parts) {
      const drained = readDrainedMessages(part);
      if (drained.length === 0) {
        if (isVisiblePart(part)) hasVisibleAbove = true;
        continue;
      }
      if (!hasVisibleAbove) continue;
      for (const entry of drained) texts.push(entry.content);
    }
  }
  return texts;
}

function splitAssistantMessage(message: ChatMessage): ChatMessage[] {
  const rows: ChatMessage[] = [];
  let segmentParts: MessagePart[] = [];
  let segmentIndex = 0;
  let followUpIndex = 0;
  // Tracked across the whole turn rather than per segment: back-to-back
  // hints leave the open segment empty, and the second one's text still has
  // to be drawn. Mirrors `hasSplittableDrainHint`, which decides the same
  // thing one message at a time.
  let hasVisibleAbove = false;

  for (const part of message.parts) {
    const drained = readDrainedMessages(part);
    if (drained.length === 0) {
      // Hints render nothing, so a text-less one simply drops through
      // instead of breaking the chain it sits in.
      segmentParts.push(part);
      if (isVisiblePart(part)) hasVisibleAbove = true;
      continue;
    }
    // Nothing drawn yet means this is the turn-start drain, whose text the
    // opening prompt already contains — splitting there would cut an empty
    // segment off the top and double the bubble.
    if (!hasVisibleAbove) continue;

    // Only cut a row when the open segment actually drew something:
    // consecutive hints must not emit empty assistant rows between bubbles.
    if (segmentParts.some(isVisiblePart)) {
      rows.push({
        ...message,
        id: `${message.id}${SEGMENT_ID_MARKER}${segmentIndex}`,
        parts: segmentParts,
      });
      segmentIndex++;
      segmentParts = [];
    }
    for (const entry of drained) {
      // Counter runs over the turn, not the segment: consecutive hints share
      // a segment index, so that alone would not keep fallback ids unique.
      rows.push(
        makeFollowUpRow(
          entry.id ?? `${message.id}-${followUpIndex}`,
          entry.content,
        ),
      );
      followUpIndex++;
    }
  }

  // The tail segment is emitted even while it is still empty (the hint is
  // the newest part of a live turn): it carries the original message id, so
  // turn stats, retry, minimap anchors and the thinking indicator — the only
  // sign that the assistant picked the follow-up up — all still land.
  rows.push({ ...message, parts: segmentParts });
  return rows;
}

function makeFollowUpRow(id: string, content: string): ChatMessage {
  return {
    id: `midturn-${id}`,
    role: "user",
    parts: [{ type: "text", text: content, state: "done" }],
  };
}

/**
 * A hint is a split point only once the turn has drawn something above it.
 * The buffer is also drained at turn *start*, where the queued text is folded
 * into the opening prompt instead of injected mid-turn: splitting there would
 * cut an empty segment off the top of the assistant message and render a
 * bubble for text the user message already contains.
 */
function hasSplittableDrainHint(message: ChatMessage): boolean {
  if (message.role !== "assistant") return false;
  let hasVisiblePartAbove = false;
  for (const part of message.parts) {
    if (readDrainedMessages(part).length > 0) {
      if (hasVisiblePartAbove) return true;
      continue;
    }
    if (isVisiblePart(part)) hasVisiblePartAbove = true;
  }
  return false;
}

/** `step-start` and bookkeeping parts draw nothing on their own. */
function isVisiblePart(part: MessagePart): boolean {
  return part.type !== "step-start" && !isBookkeepingPart(part);
}

function readDrainedMessages(part: MessagePart): DrainedMessage[] {
  if (part.type !== PENDING_DRAINED_PART_TYPE) return [];
  const data = (part as { data?: { messages?: unknown } }).data;
  if (!Array.isArray(data?.messages)) return [];
  const drained: DrainedMessage[] = [];
  for (const entry of data.messages) {
    if (typeof entry !== "object" || entry === null) continue;
    const { id, content } = entry as { id?: unknown; content?: unknown };
    if (typeof content !== "string" || content.length === 0) continue;
    drained.push({
      id: typeof id === "string" && id.length > 0 ? id : null,
      content,
    });
  }
  return drained;
}

/**
 * Turn stats are per backend turn, and a mid-turn follow-up does not start a
 * new one — so the bar reads the turn off the untouched message list rather
 * than off the split rows, where the follow-up looks like a turn boundary.
 */
export function turnMessagesForRow(
  messages: ChatMessage[],
  row: ChatMessage,
): ChatMessage[] {
  const sourceIndex = messages.findIndex((m) => m.id === row.id);
  return sourceIndex === -1 ? [row] : getTurnMessages(messages, sourceIndex);
}
