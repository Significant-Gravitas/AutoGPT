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

// Whether the chat has held a call. Its cards live under the session's own
// review id, so they are loaded even when a later run is the newest target.
export function hasHeldCall(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
): boolean {
  return messages.some((message) =>
    message.parts.some((part) => "output" in part && isHeldOutput(part.output)),
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
