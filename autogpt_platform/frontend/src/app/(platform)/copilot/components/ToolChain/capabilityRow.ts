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
  return Object.hasOwn(COPILOT_TOOL_CATALOG, key) ? key : null;
}

/** A deferred platform tool runs through `run_capability`, which returns the
 *  tool's own response unchanged: the row is the tool's row under another
 *  name, with its arguments nested one level down. Only `run_capability` is
 *  unwrapped — `describe_capability` and `validate_only` describe a call
 *  instead of making it, and `resume_capability` replays block and MCP
 *  reviews only, never a platform tool. A failed call stays as it is too:
 *  several tool cards draw from the input alone and would hide the error. */
export function capabilityTargetRow(row: ChainRow): ChainRow {
  if (row.tool !== "run_capability") return row;
  const call = asObject(row.input);
  if (!call || call.validate_only === true) return row;
  const type = asObject(row.output)?.type;
  if (type === "capability_details" || type === "error") return row;
  const tool = platformToolName(str(call, "id") ?? "");
  if (!tool || CAPABILITY_TOOLS.has(tool)) return row;
  return { ...row, tool, input: asObject(call.input) ?? {} };
}

/** The part-level sibling of `capabilityTargetRow`, for cards that are lifted
 *  out of the chain by tool name. It reads the input alone: an approval card
 *  has to be on screen while the call is still pending. */
export function capabilityTargetToolName(
  type: string,
  input: unknown,
): string | null {
  if (type !== "tool-run_capability") return null;
  const call = asObject(input);
  if (!call || call.validate_only === true) return null;
  const tool = platformToolName(str(call, "id") ?? "");
  return tool && !CAPABILITY_TOOLS.has(tool) ? tool : null;
}
