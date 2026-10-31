import type { UIDataTypes, UIMessage, UITools } from "ai";
import type { VirtualItem } from "@tanstack/react-virtual";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

export interface TranscriptRow {
  message: ChatMessage;
  /** Position in the full list of rendered rows, which also holds the rows
   *  the transcript leaves out (a held call's result row). */
  rowIndex: number;
}

type TranscriptSegment =
  | { kind: "row"; index: number }
  | { kind: "gap"; afterIndex: number; height: number };

/** Rows kept mounted above and below the visible window. */
export const OVERSCAN_ROWS = 6;

/** Most rows from the newest user message to the end that stay mounted
 *  wherever the user scrolls, so the live turn and `TailSpacer`'s anchor
 *  never unmount. */
const MAX_PINNED_TAIL_ROWS = 8;

const TEXT_BUCKETS: { maxChars: number; height: number }[] = [
  { maxChars: 200, height: 96 },
  { maxChars: 1_000, height: 240 },
  { maxChars: 4_000, height: 560 },
];
const HUGE_ROW_HEIGHT = 1_200;
const TOOL_PART_HEIGHT = 44;
const MAX_COUNTED_TOOL_PARTS = 10;

/**
 * Height guess for a row that has not been measured yet. Bucketed by text
 * length so a handful of very tall replies cannot inflate the guess for
 * short ones: when a short row measures far below its guess, the total
 * shrinks and pulls the next row into the window in the same layout pass,
 * and a long chain of those can hit React's update-depth limit.
 */
export function estimateRowHeight(message: ChatMessage): number {
  let chars = 0;
  let toolParts = 0;
  for (const part of message.parts) {
    if (part.type === "text" || part.type === "reasoning") {
      chars += part.text.length;
    } else if (part.type.startsWith("tool-")) {
      toolParts += 1;
    }
  }
  const bucket = TEXT_BUCKETS.find((entry) => chars <= entry.maxChars);
  const textHeight = bucket ? bucket.height : HUGE_ROW_HEIGHT;
  return (
    textHeight + Math.min(toolParts, MAX_COUNTED_TOOL_PARTS) * TOOL_PART_HEIGHT
  );
}

/** Rows that stay mounted however far the user scrolls: the newest user row,
 *  which `TailSpacer` measures from, and the rows of its turn up to the last
 *  `MAX_PINNED_TAIL_ROWS`. */
function pinnedIndexes(
  rows: TranscriptRow[],
  lastUserMessageID: string | null,
): number[] {
  if (!lastUserMessageID) return [];
  const userIndex = rows.findLastIndex(
    (row) => row.message.id === lastUserMessageID,
  );
  if (userIndex < 0) return [];
  const pinned = [userIndex];
  const tailStart = Math.max(userIndex + 1, rows.length - MAX_PINNED_TAIL_ROWS);
  for (let i = tailStart; i < rows.length; i++) pinned.push(i);
  return pinned;
}

/** The window's row indexes plus the pinned ones, ascending and unique. */
function mergeRenderedIndexes(
  windowIndexes: number[],
  pinned: number[],
): number[] {
  const merged = new Set([...windowIndexes, ...pinned]);
  return Array.from(merged).sort((a, b) => a - b);
}

/**
 * Lays the rendered rows out in normal flow: the list's own padding stands in
 * for the rows above and below, and a spacer fills any hole between two
 * rendered rows (the window and the pinned tail). Each spacer sits between two
 * flex gaps, so it is two gaps shorter than the distance it covers.
 */
function layoutSegments({
  indexes,
  measurements,
  gap,
}: {
  indexes: number[];
  measurements: VirtualItem[];
  gap: number;
}): TranscriptSegment[] {
  const segments: TranscriptSegment[] = [];
  indexes.forEach((index, position) => {
    const previous = indexes[position - 1];
    if (previous !== undefined && index > previous + 1) {
      const height =
        measurements[index].start - measurements[previous].end - 2 * gap;
      segments.push({
        kind: "gap",
        afterIndex: previous,
        height: Math.max(0, height),
      });
    }
    segments.push({ kind: "row", index });
  });
  return segments;
}

/** Padding that stands in for the unmounted rows above and below. */
function listPadding({
  indexes,
  measurements,
  scrollMargin,
  totalSize,
}: {
  indexes: number[];
  measurements: VirtualItem[];
  scrollMargin: number;
  totalSize: number;
}): { top: number; bottom: number } {
  const first = measurements[indexes[0]];
  const last = measurements[indexes[indexes.length - 1]];
  if (!first || !last) return { top: 0, bottom: 0 };
  return {
    top: Math.max(0, first.start - scrollMargin),
    bottom: Math.max(0, totalSize - (last.end - scrollMargin)),
  };
}

/** The rows to mount and how to lay them out. Without a window (no
 *  measurable viewport) every row renders with no padding. */
export function windowLayout({
  rows,
  windowIndexes,
  isWindowed,
  lastUserMessageID,
  measurements,
  gap,
  scrollMargin,
  totalSize,
}: {
  rows: TranscriptRow[];
  windowIndexes: number[];
  isWindowed: boolean;
  lastUserMessageID: string | null;
  measurements: VirtualItem[];
  gap: number;
  scrollMargin: number;
  totalSize: number;
}) {
  if (!isWindowed) {
    const indexes = rows.map((_, index) => index);
    return {
      indexes,
      segments: indexes.map((index) => ({ kind: "row" as const, index })),
      padding: { top: 0, bottom: 0 },
    };
  }
  const indexes = mergeRenderedIndexes(
    windowIndexes,
    pinnedIndexes(rows, lastUserMessageID),
  );
  return {
    indexes,
    segments: layoutSegments({ indexes, measurements, gap }),
    padding: listPadding({ indexes, measurements, scrollMargin, totalSize }),
  };
}
