import type { UIDataTypes, UIMessage, UITools } from "ai";
import { TOOL_PART_PREFIX } from "../JobStatsBar/constants";
import {
  isBookkeepingPart,
  PENDING_DRAINED_PART_TYPE,
} from "../../messageParts";
import { TURN_PAUSED_PART_TYPE } from "../../turnPause";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

/** What the newest assistant message is doing, as far as the thinking
 *  indicator is concerned.
 *  - `ended`: the turn has said how it ends (a pause); nothing to wait on.
 *  - `working`: something on screen shows the work itself (a tool row
 *    filling in or running, a live reasoning row).
 *  - `writing`: text that may still be growing; `signature` changes whenever
 *    it grows, so a signature that holds still means the text has stalled.
 *  - `quiet`: nothing is arriving. `midTurn` says the turn has already shown
 *    something, so a short gap between steps need not draw the indicator. */
export type TurnActivity =
  | { kind: "ended" }
  | { kind: "working" }
  | { kind: "writing"; signature: string }
  | { kind: "quiet"; midTurn: boolean };

const QUIET_NOW: TurnActivity = { kind: "quiet", midTurn: false };

export function getTurnActivity(
  message: ChatMessage | undefined,
  status: string,
): TurnActivity {
  // A new request is out and nothing of its reply has arrived yet.
  if (status === "submitted" || message?.role !== "assistant") {
    return QUIET_NOW;
  }
  const { parts } = message;
  // Bookkeeping parts are not content. A drain hint newer than the last
  // content part is the exception: the follow-up just landed and nothing has
  // been produced past it, so the indicator is the only sign the assistant
  // picked the follow-up up.
  let lastIndex = parts.length - 1;
  while (lastIndex >= 0 && isBookkeepingPart(parts[lastIndex])) {
    if (parts[lastIndex].type === PENDING_DRAINED_PART_TYPE) return QUIET_NOW;
    lastIndex--;
  }
  if (lastIndex < 0) return QUIET_NOW;
  return getPartActivity(parts[lastIndex], lastIndex);
}

function getPartActivity(
  part: ChatMessage["parts"][number],
  index: number,
): TurnActivity {
  if (part.type === TURN_PAUSED_PART_TYPE) return { kind: "ended" };
  if (
    part.type.startsWith(TOOL_PART_PREFIX) &&
    "state" in part &&
    (part.state === "input-streaming" || part.state === "input-available")
  ) {
    return { kind: "working" };
  }
  if (part.type === "reasoning" && part.state === "streaming") {
    return { kind: "working" };
  }
  if (
    (part.type === "text" || part.type === "reasoning") &&
    part.text.trim().length > 0
  ) {
    return { kind: "writing", signature: `${index}:${part.text.length}` };
  }
  return { kind: "quiet", midTurn: true };
}
