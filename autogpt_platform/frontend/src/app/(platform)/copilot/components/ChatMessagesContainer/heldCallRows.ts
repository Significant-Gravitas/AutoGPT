import type { UIDataTypes, UIMessage, UITools } from "ai";

// Rows the server writes for a held call: the one that starts the turn when a
// card is answered, and the call's late result, which the reply narrates.
export function getHeldCallRowKind(
  metadata: unknown,
): "answered" | "result" | null {
  if (!metadata || typeof metadata !== "object") return null;
  if ("held_call" in metadata) return "result";
  if ("held_calls_answered" in metadata) return "answered";
  return null;
}

export function isHeldCallRow(message: { metadata?: unknown }): boolean {
  return getHeldCallRowKind(message.metadata) !== null;
}

// How many calls on screen were held. A new one means a new card to fetch;
// an old one says nothing about whether its card is still open.
export function countHeldCalls(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
): number {
  return messages.reduce(
    (count, message) =>
      count +
      message.parts.filter(
        (part) => "output" in part && heldReviewId(part.output) !== null,
      ).length,
    0,
  );
}

function heldReviewId(output: unknown): string | null {
  let value = output;
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return null;
    }
  }
  if (
    !value ||
    typeof value !== "object" ||
    (value as { type?: unknown }).type !== "approval_required"
  )
    return null;
  const reviewId = (value as { review_id?: unknown }).review_id;
  return typeof reviewId === "string" ? reviewId : null;
}

export type HeldOutcomeKind =
  | "approved"
  | "rejected"
  | "expired"
  | "closed"
  | "unknown";

export interface HeldOutcome {
  outcome: HeldOutcomeKind;
  // The run's own output, as the tool would have returned it directly.
  output: unknown;
}

const OUTCOMES = new Set<string>([
  "approved",
  "rejected",
  "expired",
  "closed",
  "unknown",
]);
// Anchored at the end: the output itself may contain the closing tag.
const RESULT_RE =
  /<held_call_result[^>]*>\n?([\s\S]*?)\n?<\/held_call_result>\s*$/;

export function getHeldOutcomes(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
): Map<string, HeldOutcome> {
  const outcomes = new Map<string, HeldOutcome>();
  // Held calls not yet paired with a result, oldest first per review id.
  const unanswered = new Map<string, string[]>();
  for (const message of messages) {
    const held = heldCallMetadata(message.metadata);
    if (!held) {
      for (const part of message.parts) {
        if (!("output" in part) || !("toolCallId" in part)) continue;
        const reviewId = heldReviewId(part.output);
        if (reviewId === null) continue;
        unanswered.set(reviewId, [
          ...(unanswered.get(reviewId) ?? []),
          part.toolCallId,
        ]);
      }
      continue;
    }
    const text = message.parts
      .map((part) => (part.type === "text" ? part.text : ""))
      .join("");
    const body = RESULT_RE.exec(text)?.[1] ?? "";
    const outcome: HeldOutcome = {
      // Rows persisted before the outcome was recorded say "Nothing ran" when refused.
      outcome:
        held.outcome ??
        (body.startsWith("Nothing ran") ? "closed" : "approved"),
      output: parseOutput(body),
    };
    outcomes.set(held.toolCallId, outcome);
    // The SDK engine records a stand-in id, not the model's; the review id
    // pairs the result with the earliest held call still waiting on it.
    const waiting = held.reviewId ? unanswered.get(held.reviewId) : undefined;
    if (!waiting?.length) continue;
    const index = waiting.includes(held.toolCallId)
      ? waiting.indexOf(held.toolCallId)
      : 0;
    outcomes.set(waiting[index], outcome);
    waiting.splice(index, 1);
  }
  return outcomes;
}

function heldCallMetadata(metadata: unknown) {
  if (!metadata || typeof metadata !== "object") return null;
  const held = (metadata as { held_call?: unknown }).held_call;
  if (!held || typeof held !== "object") return null;
  const { tool_call_id, review_id, outcome } = held as Record<string, unknown>;
  if (typeof tool_call_id !== "string" || !tool_call_id) return null;
  return {
    toolCallId: tool_call_id,
    reviewId: typeof review_id === "string" ? review_id : null,
    outcome:
      typeof outcome === "string" && OUTCOMES.has(outcome)
        ? (outcome as HeldOutcomeKind)
        : null,
  };
}

function parseOutput(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}
