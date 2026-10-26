import type { UIDataTypes, UIMessage, UITools } from "ai";
import { parseSpecialMarkers } from "../../helpers/messageMarkers";
import { TURN_PAUSED_PART_TYPE } from "../../turnPause";
import { isHeldCallOutput } from "../ChatMessagesContainer/heldCallRows";
import { getTurnMessages } from "../ChatMessagesContainer/helpers";
import { isToolCallPending } from "../ToolChain/helpers";
import { formatElapsed } from "./formatElapsed";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

export type TurnOutcome = "done" | "stopped" | "unsettled";

const STOP_MARKER_TEXTS = new Set([
  "operation cancelled",
  "execution stopped by user",
]);

/**
 * How a settled turn ended. "Done" is claimed only for a turn that ran to
 * its end: a stop, an error or system note, a pause, a call still waiting
 * for the user, or a call with no result all leave it "unsettled".
 */
export function getTurnOutcome(turnMessages: ChatMessage[]): TurnOutcome {
  const parts = turnMessages
    .filter((message) => message.role === "assistant")
    .flatMap((message) => message.parts);
  if (parts.length === 0) return "unsettled";

  const lastText = parts.findLast((part) => part.type === "text");
  if (lastText) {
    const { markerType, markerText } = parseSpecialMarkers(lastText.text);
    if (markerType !== null) {
      return STOP_MARKER_TEXTS.has(markerText.toLowerCase())
        ? "stopped"
        : "unsettled";
    }
  }

  const waiting = parts.some(
    (part) =>
      part.type === TURN_PAUSED_PART_TYPE ||
      isToolCallPending(part) ||
      ("output" in part && isHeldCallOutput(part.output)),
  );
  return waiting ? "unsettled" : "done";
}

export function getTurnEndLabel(
  outcome: TurnOutcome,
  seconds: number | null,
): string | null {
  const time = seconds === null ? null : formatElapsed(seconds);
  if (outcome === "done") return time ? `Done in ${time}` : "Done";
  if (outcome === "stopped") return time ? `Stopped after ${time}` : "Stopped";
  return time ? `Thought for ${time}` : null;
}

/** What a screen reader hears when a turn it watched streaming ends. */
export function getLiveTurnEndAnnouncement(messages: ChatMessage[]): string {
  const last = messages.length - 1;
  if (messages[last]?.role !== "assistant") return "";
  const outcome = getTurnOutcome(getTurnMessages(messages, last));
  return outcome === "done" ? "Reply finished" : "";
}
