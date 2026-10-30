import type { UIDataTypes, UIMessage, UITools } from "ai";
import { parseSpecialMarkers } from "../../helpers/messageMarkers";
import { TURN_PAUSED_PART_TYPE } from "../../turnPause";
import { getTurnMessages } from "../ChatMessagesContainer/helpers";
import { isToolCallPending } from "../ToolChain/helpers";
import { asObject } from "../ToolChain/resultHelpers";
import { formatElapsed } from "./formatElapsed";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

export type TurnOutcome = "done" | "stopped" | "unsettled" | "settling";

const STOP_MARKER_TEXTS = new Set([
  "operation cancelled",
  "execution stopped by user",
]);

// A held action is only ever answered in a later turn, so an approval or
// review card left in the turn keeps it open whatever the model did next.

// Results that leave the turn waiting on the user (a sign-in or setup)
// unless the model went on to other work after them.
const WAITING_RESULT_TYPES = new Set([
  "setup_requirements",
  "need_login",
  "trigger_config_required",
]);

type Part = ChatMessage["parts"][number];

function isToolPart(part: Part) {
  return part.type.startsWith("tool-");
}

function resultType(part: Part) {
  if (!isToolPart(part) || !("output" in part)) return null;
  const type = asObject(part.output)?.type;
  return typeof type === "string" ? type : null;
}

// A refusal with no review behind it (declined, already used) waits on no one.
function isHeldResult(part: Part) {
  if (!isToolPart(part) || !("output" in part)) return false;
  const output = asObject(part.output);
  return (
    output?.type === "review_required" ||
    (output?.type === "approval_required" &&
      typeof output.review_id === "string")
  );
}

function isWaitingResult(part: Part) {
  return WAITING_RESULT_TYPES.has(resultType(part) ?? "");
}

function isStillStreaming(part: Part) {
  return (
    (part.type === "text" || part.type === "reasoning") &&
    part.state === "streaming"
  );
}

/**
 * How a settled turn ended. "Done" is claimed only for a turn that ran to
 * its end. A stop, an error or system note, a call still running, and a
 * last step that failed with nothing after it leave it "unsettled". So do a
 * pause or a result waiting on the user, unless the turn went on to run
 * more work after it (a continuation after "Keep going" or an approval).
 * Text still marked streaming was cut off (a stream that gave up), since a
 * finished part is always marked done.
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

  if (
    parts.some(isToolCallPending) ||
    parts.some(isStillStreaming) ||
    parts.some(isHeldResult)
  ) {
    return "unsettled";
  }

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
 * really ended; earlier turns keep the plain "Thought for" line. A finished
 * turn whose stream is still settling shows just its time, so the line
 * doesn't read "Thought for" and then switch to "Done"; one that can't settle
 * (an error, a host that doesn't report it) reads "Thought for".
 */
export function getDisplayedOutcome(
  turnMessages: ChatMessage[],
  isLatest: boolean,
  isSettled: boolean,
  isSettling: boolean,
): TurnOutcome {
  if (!isLatest) return "unsettled";
  const outcome = getTurnOutcome(turnMessages);
  if (outcome !== "done" || isSettled) return outcome;
  return isSettling ? "settling" : "unsettled";
}

export function getTurnEndLabel(
  outcome: TurnOutcome,
  seconds: number | null,
): string | null {
  const time = seconds === null ? null : formatElapsed(seconds);
  if (outcome === "done") return time ? `Done in ${time}` : "Done";
  if (outcome === "stopped") return time ? `Stopped after ${time}` : "Stopped";
  if (outcome === "settling") return time;
  return time ? `Thought for ${time}` : null;
}

/** What a screen reader hears when a turn it watched streaming ends. */
export function getLiveTurnEndAnnouncement(messages: ChatMessage[]): string {
  const last = messages.length - 1;
  if (messages[last]?.role !== "assistant") return "";
  const outcome = getTurnOutcome(getTurnMessages(messages, last));
  return outcome === "done" ? "Reply finished" : "";
}
