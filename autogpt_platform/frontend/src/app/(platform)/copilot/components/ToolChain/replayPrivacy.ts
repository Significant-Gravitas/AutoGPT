import { capabilityTargetRow } from "./capabilityRow";
import type { ChainRow } from "./helpers";

// Tools that hand back what a block returned: a Google block's output is the
// user's Gmail, Drive, Docs, Sheets or Calendar data. Every block run is
// hidden, not only Google's, because a large output reaches the chat as a
// truncated preview that no longer reliably says which provider produced it.
const BLOCK_RUN_TOOLS = new Set([
  "run_block",
  "continue_run_block",
  "run_capability",
  "resume_capability",
]);

export function hidesOutputFromReplay(row: ChainRow): boolean {
  const tool = capabilityTargetRow(row).tool;
  return tool !== undefined && BLOCK_RUN_TOOLS.has(tool);
}
