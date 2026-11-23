import type { ChainRow } from "./helpers";

export const SUB_SESSION_TOOLS = new Set([
  "run_sub_session",
  "delegate_to_expert",
  "handoff_to_expert",
  "get_sub_session_result",
]);

export function isLiveSubSessionRow(row: ChainRow): boolean {
  if (!row.tool || !SUB_SESSION_TOOLS.has(row.tool)) return false;
  // A blocking delegate has no output while the teammate works — that IS
  // the live window, covered by the pending card built from the input.
  // Result polls render no pending card (see ToolResult), so an outputless
  // poll row has nothing to open.
  if (row.output === undefined)
    return row.state === "running" && row.tool !== "get_sub_session_result";
  if (!row.output || typeof row.output !== "object") return false;
  const status = (row.output as { status?: unknown }).status;
  return (
    typeof status === "string" &&
    ["running", "queued"].includes(status.toLowerCase())
  );
}
