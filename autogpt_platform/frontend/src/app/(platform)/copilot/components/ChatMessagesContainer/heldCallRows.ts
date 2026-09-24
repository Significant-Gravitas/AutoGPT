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
        (part) => "output" in part && isHeldOutput(part.output),
      ).length,
    0,
  );
}

function isHeldOutput(output: unknown): boolean {
  let value = output;
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return false;
    }
  }
  return (
    !!value &&
    typeof value === "object" &&
    (value as { type?: unknown }).type === "approval_required" &&
    typeof (value as { review_id?: unknown }).review_id === "string"
  );
}
