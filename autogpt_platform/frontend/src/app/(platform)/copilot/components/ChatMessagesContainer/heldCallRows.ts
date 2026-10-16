import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { cardArguments, fallbackAsk } from "../ApprovalQueue/helpers";
import { heldAskText, heldToolName, isHeldReadId } from "../ToolChain/heldRow";

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

// What a wake row says: the cards its turn delivered, by their own headlines,
// and what each answer sent on, from the result row's recorded outcome.
export function answeredLabel(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
  wakeIndex: number,
): string {
  const asks = new Map<string, string>();
  for (const message of messages.slice(0, wakeIndex)) {
    for (const part of message.parts) {
      if (!("output" in part)) continue;
      const card = heldCardOutput(part.output);
      const reviewId = card ? str(card.review_id) : null;
      if (!card || !reviewId) continue;
      asks.set(reviewId, heldAskText(card, heldToolName(card, partTool(part))));
    }
  }
  const byOutcome = new Map<AnswerKind | null, string[]>();
  for (const message of messages.slice(wakeIndex + 1)) {
    if (getHeldCallRowKind(message.metadata) !== "result") break;
    const held = (message.metadata as { held_call: Record<string, unknown> })
      .held_call;
    const tool = str(held?.tool_name);
    const name =
      asks.get(str(held?.review_id) ?? "") ?? (tool && fallbackAsk(tool));
    if (!name) continue;
    const outcome = str(held?.outcome);
    const recorded =
      outcome && OUTCOMES.has(outcome) ? (outcome as HeldOutcomeKind) : null;
    // Only a read closes after an identical re-read already delivered it.
    const kind =
      recorded === "closed" && isHeldReadId(str(held?.review_id))
        ? "closed-read"
        : recorded;
    byOutcome.set(kind, [...(byOutcome.get(kind) ?? []), name]);
  }
  if (!byOutcome.size) return "Approval answered";
  const clauses = [...byOutcome].map(([outcome, names]) =>
    [`${names.join(", ")}.`, outcome && sentOn(outcome, names.length > 1)]
      .filter(Boolean)
      .join(" "),
  );
  return `Approval answered: ${clauses.join(" ")}`;
}

// Every answer writes a result row, but only an approval's carries a result.
function sentOn(outcome: AnswerKind, several: boolean): string {
  switch (outcome) {
    case "approved":
      return `${several ? "Results" : "Result"} sent to ${AUTOPILOT_NAME}.`;
    case "rejected":
      return `Declined; nothing was sent to ${AUTOPILOT_NAME}.`;
    case "expired":
      return `Expired; nothing was sent to ${AUTOPILOT_NAME}.`;
    case "closed":
      return "Closed; nothing ran.";
    case "closed-read":
      return `Closed; nothing was released, or ${AUTOPILOT_NAME} already had it.`;
    case "unknown":
      return "Unclear; check before relying on it.";
  }
}

function heldReviewId(output: unknown): string | null {
  const reviewId = heldCardOutput(output)?.review_id;
  return typeof reviewId === "string" ? reviewId : null;
}

function heldCardOutput(output: unknown): Record<string, unknown> | null {
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
  return value as Record<string, unknown>;
}

function partTool(part: { type: string; toolName?: unknown }): string {
  if (part.type === "dynamic-tool") return str(part.toolName) ?? "";
  return part.type.startsWith("tool-") ? part.type.slice(5) : "";
}

function str(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}

export type HeldOutcomeKind =
  | "approved"
  | "rejected"
  | "expired"
  | "closed"
  | "unknown";

type AnswerKind = HeldOutcomeKind | "closed-read";

export interface HeldOutcome {
  outcome: HeldOutcomeKind;
  // The run's own output, as the tool would have returned it directly.
  output: unknown;
  // The arguments as the card showed them; the card itself is gone by now.
  card?: HeldCard;
}

export type HeldCard = ReturnType<typeof cardArguments>;

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
      card: held.card,
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
  const card = cardArguments(held as Record<string, unknown>);
  return {
    toolCallId: tool_call_id,
    reviewId: typeof review_id === "string" ? review_id : null,
    outcome:
      typeof outcome === "string" && OUTCOMES.has(outcome)
        ? (outcome as HeldOutcomeKind)
        : null,
    card: Object.keys(card.args).length > 0 ? card : undefined,
  };
}

function parseOutput(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}
