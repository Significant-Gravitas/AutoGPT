import type { UIDataTypes, UIMessage, UITools } from "ai";
import { TOOL_PART_PREFIX } from "../JobStatsBar/constants";
import { countHeldCalls } from "./heldCallRows";

// The tools whose call can stop on a block review. The chat cannot always tell
// a review from the output (a large one used to arrive digested), so every one
// that returns fetches the chat's cards again.
const BLOCK_RUN_TOOLS = new Set([
  "run_block",
  "run_capability",
  "continue_run_block",
  "resume_capability",
]);

/** Changes whenever a call on screen may have raised a card: a held call, or a
 *  block run that returned. */
export function reviewRefetchKey(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
): number {
  const blockRuns = messages.reduce(
    (count, message) => count + message.parts.filter(isFinishedBlockRun).length,
    0,
  );
  return countHeldCalls(messages) + blockRuns;
}

function isFinishedBlockRun(
  part: UIMessage<unknown, UIDataTypes, UITools>["parts"][number],
) {
  return (
    part.type.startsWith(TOOL_PART_PREFIX) &&
    BLOCK_RUN_TOOLS.has(part.type.slice(TOOL_PART_PREFIX.length)) &&
    "state" in part &&
    part.state === "output-available"
  );
}
