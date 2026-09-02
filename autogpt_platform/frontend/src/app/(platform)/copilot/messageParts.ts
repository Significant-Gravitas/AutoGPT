import type { UIMessage } from "ai";

import { COMPACTION_DATA_PART_TYPE } from "./components/CompactionCard/helpers";
import { DREAM_OPERATIONS_PART_TYPE } from "./dreamOperations";

/** Transient copy for the Thinking indicator — never rendered inline. */
export const STATUS_PART_TYPE = "data-status";
/** Legacy resume metadata. */
const CURSOR_PART_TYPE = "data-cursor";

/**
 * Stream bookkeeping: parts the backend sends to drive chrome (indicators,
 * resume cursors, progress rows) rather than message content.
 *
 * This set is the single list. Every new transient `data-*` part must be
 * added here and nowhere else — the filters used to enumerate them one call
 * site at a time, and missing one is a silent UI bug (a stray part counts as
 * "real content", hides the Thinking indicator, splits a tool chain, or
 * kills a live progress bar) rather than a type error.
 */
const BOOKKEEPING_PART_TYPES: ReadonlySet<string> = new Set([
  CURSOR_PART_TYPE,
  STATUS_PART_TYPE,
  DREAM_OPERATIONS_PART_TYPE,
  COMPACTION_DATA_PART_TYPE,
]);

export function isBookkeepingPart(part: { type: string }): boolean {
  return BOOKKEEPING_PART_TYPES.has(part.type);
}

/**
 * Surface the latest backend-emitted status message for the trailing assistant
 * message, if that status has not already been invalidated by newer visible
 * parts. Used to show progress during restore/replay before answer text lands.
 */
export function getLatestAssistantStatusMessage(
  messages: UIMessage[],
): string | null {
  const last = messages[messages.length - 1];
  if (last?.role !== "assistant") return null;
  for (let i = last.parts.length - 1; i >= 0; i--) {
    const part = last.parts[i];
    if (part.type === STATUS_PART_TYPE) {
      const data = (part as { data?: { message?: unknown } }).data;
      return typeof data?.message === "string" ? data.message : null;
    }
    // Other bookkeeping lands after the status all the time (a compaction
    // starts, a cursor checkpoints) without the model having said anything
    // new — reading through it keeps a still-relevant status on screen.
    if (isBookkeepingPart(part)) continue;
    // Anything else = the model has produced output past the status.
    return null;
  }
  return null;
}
