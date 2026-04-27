import type { UIMessage } from "ai";

/**
 * Build a user-bubble `UIMessage` that represents pending chips which the
 * backend has just drained. The prefix identifies the promotion path —
 * `auto-continue` bubbles sit right before the auto-continue assistant,
 * while `midturn` bubbles sit at the tail of the visible chat while the
 * stream is still going. Force-hydrate replaces both flavours with the
 * real DB row once the stream ends.
 */
export function makePromotedUserBubble(
  texts: string[],
  prefix: "auto-continue" | "midturn",
  suffix: string,
): UIMessage {
  return {
    id: `promoted-${prefix}-${suffix}`,
    role: "user" as const,
    parts: [
      {
        type: "text" as const,
        text: texts.join("\n\n"),
        state: "done" as const,
      },
    ],
  };
}
