import type { UIMessage } from "ai";

/**
 * Build a user-bubble `UIMessage` representing a single pending chip the
 * backend has just drained. One chip → one bubble (preserves cardinality
 * across the chip→drain→promote lifecycle). The prefix identifies the
 * promotion path — `auto-continue` bubbles sit right before the
 * auto-continue assistant, while `midturn` bubbles sit at the tail of
 * the visible chat while the stream is still going. Force-hydrate replaces
 * both flavours with the real DB row once the stream ends.
 */
export function makePromotedUserBubble(
  text: string,
  prefix: "auto-continue" | "midturn",
  suffix: string,
): UIMessage {
  return {
    id: promotedBubbleId(prefix, suffix),
    role: "user" as const,
    parts: [
      {
        type: "text" as const,
        text,
        state: "done" as const,
      },
    ],
  };
}

/** The id `makePromotedUserBubble` gives a bubble; it does not depend on the text. */
export function promotedBubbleId(
  prefix: "auto-continue" | "midturn",
  suffix: string,
): string {
  return `promoted-${prefix}-${suffix}`;
}
