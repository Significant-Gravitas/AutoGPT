import type { UIDataTypes, UIMessage, UITools } from "ai";
import { parseSpecialMarkers } from "../../helpers/messageMarkers";
import { TURN_PAUSED_PART_TYPE } from "../../turnPause";
import { getTurnMessages } from "../ChatMessagesContainer/helpers";
import { isToolCallPending } from "../ToolChain/helpers";
import { asObject } from "../ToolChain/resultHelpers";
import { formatElapsed } from "./formatElapsed";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

export type TurnOutcome = "done" | "stopped" | "unsettled";

const STOP_MARKER_TEXTS = new Set([
  "operation cancelled",
  "execution stopped by user",
]);

// Results that leave the turn waiting on the user (a review, an approval, a
// sign-in or setup) rather than finished.
const WAITING_RESULT_TYPES = new Set([
  "approval_required",
  "review_required",
  "setup_requirements",
  "need_login",
  "trigger_config_required",
]);

type Part = ChatMessage["parts"][number];

function isToolPart(part: Part) {
  return part.type.startsWith("tool-");
}

function isWaitingResult(part: Part) {
  if (!isToolPart(part) || !("output" in part)) return false;
  const type = asObject(part.output)?.type;
  return typeof type === "string" && WAITING_RESULT_TYPES.has(type);
}

/**
 * How a settled turn ended. "Done" is claimed only for a turn that ran to
 * its end. A stop, an error or system note, a call still running, and a
 * last step that failed with nothing after it leave it "unsettled". So do a
 * pause or a result waiting on the user, unless the turn went on to run
 * more work after it (a continuation after "Keep going" or an approval).
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

  if (parts.some(isToolCallPending)) return "unsettled";

  const lastWork = parts.findLast(
    (part) => isToolPart(part) || (part.type === "text" && part.text.trim()),
  );
  if (lastWork && isToolPart(lastWork) && "state" in lastWork) {
    if (lastWork.state === "output-error") return "unsettled";
  }

  const lastPause = parts.findLastIndex(
    (part) => part.type === TURN_PAUSED_PART_TYPE,
  );
  if (lastPause !== -1 && parts.indexOf(lastWork as Part) < lastPause) {
    return "unsettled";
  }

  const lastWaiting = parts.findLastIndex(isWaitingResult);
  const lastTool = parts.findLastIndex(isToolPart);
  return lastWaiting !== -1 && lastWaiting === lastTool ? "unsettled" : "done";
}

/**
 * Only the latest turn says how it ended, and only once its stream has
 * really ended; earlier turns keep the plain "Thought for" line.
 */
export function getDisplayedOutcome(
  turnMessages: ChatMessage[],
  isLatest: boolean,
  isSettled: boolean,
): TurnOutcome {
  if (!isLatest) return "unsettled";
  const outcome = getTurnOutcome(turnMessages);
  return outcome === "done" && !isSettled ? "unsettled" : outcome;
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
