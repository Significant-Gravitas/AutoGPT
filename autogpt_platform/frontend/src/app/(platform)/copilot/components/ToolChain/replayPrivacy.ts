import { capabilityTargetRow } from "./capabilityRow";
import type { ChainRow } from "./helpers";

// Tools that hand back what a block returned: a Google block's output is the
// user's Gmail, Drive, Docs, Sheets or Calendar data. Every block run is
// hidden, not only Google's, because a large output reaches the chat as a
// truncated preview that no longer reliably says which provider produced it.
// `read_workspace_file` reads a large output back from `tool-outputs/`; every
// read is hidden, because a read by `file_id` names no path, and a large read
// reaches the chat as a truncated preview that no longer carries one.
const HIDDEN_OUTPUT_TOOLS = new Set([
  "run_block",
  "continue_run_block",
  "run_capability",
  "resume_capability",
  "read_workspace_file",
]);

export function hidesOutputFromReplay(row: ChainRow): boolean {
  const tool = capabilityTargetRow(row).tool;
  return tool !== undefined && HIDDEN_OUTPUT_TOOLS.has(tool);
}
