import type { ChainRow } from "./helpers";
import { asObject, str } from "./resultHelpers";
import { COPILOT_TOOL_CATALOG } from "./toolCatalog";

const TOOL_ID_PREFIX = "tool:";

const CAPABILITY_TOOLS = new Set([
  "find_capability",
  "describe_capability",
  "run_capability",
  "resume_capability",
]);

function platformToolName(id: string): string | null {
  const key = id.trim();
  if (key.toLowerCase().startsWith(TOOL_ID_PREFIX)) {
    return key.slice(TOOL_ID_PREFIX.length).trim() || null;
  }
  return key in COPILOT_TOOL_CATALOG ? key : null;
}

/** A deferred platform tool runs through `run_capability`, which returns the
 *  tool's own response unchanged: the row is the tool's row under another
 *  name, with its arguments nested one level down. Only `run_capability` is
 *  unwrapped — `describe_capability` and `validate_only` describe a call
 *  instead of making it, and `resume_capability` replays block and MCP
 *  reviews only, never a platform tool. */
export function capabilityTargetRow(row: ChainRow): ChainRow {
  if (row.tool !== "run_capability") return row;
  const call = asObject(row.input);
  if (!call || call.validate_only === true) return row;
  if (asObject(row.output)?.type === "capability_details") return row;
  const tool = platformToolName(str(call, "id") ?? "");
  if (!tool || CAPABILITY_TOOLS.has(tool)) return row;
  return { ...row, tool, input: asObject(call.input) ?? {} };
}
