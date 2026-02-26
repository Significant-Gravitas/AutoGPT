import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { WarningDiamondIcon, PlugsConnectedIcon } from "@phosphor-icons/react";
import type { ToolUIPart } from "ai";
import { OrbitLoader } from "../../components/OrbitLoader/OrbitLoader";

// ------------------------------------------------------------------ //
//  Response type literals
// ------------------------------------------------------------------ //

const MCP_TOOLS_DISCOVERED = "mcp_tools_discovered";
const MCP_TOOL_OUTPUT = "mcp_tool_output";
const SETUP_REQUIREMENTS = "setup_requirements";
const ERROR = "error";

const RUN_MCP_TOOL_OUTPUT_TYPES = new Set<string>([
  MCP_TOOLS_DISCOVERED,
  MCP_TOOL_OUTPUT,
  SETUP_REQUIREMENTS,
  ERROR,
]);

// ------------------------------------------------------------------ //
//  Inline types (avoids waiting for OpenAPI codegen)
// ------------------------------------------------------------------ //

export interface MCPToolInfo {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

export interface MCPToolsDiscoveredOutput {
  type: typeof MCP_TOOLS_DISCOVERED;
  message: string;
  server_url: string;
  tools: MCPToolInfo[];
  session_id?: string | null;
}

export interface MCPToolOutputResult {
  type: typeof MCP_TOOL_OUTPUT;
  message: string;
  server_url: string;
  tool_name: string;
  result: unknown;
  success: boolean;
  session_id?: string | null;
}

export interface MCPErrorOutput {
  type: typeof ERROR;
  message: string;
  error?: string | null;
  session_id?: string | null;
}

export type RunMCPToolOutput =
  | MCPToolsDiscoveredOutput
  | MCPToolOutputResult
  | SetupRequirementsResponse
  | MCPErrorOutput;

// ------------------------------------------------------------------ //
//  Type guards
// ------------------------------------------------------------------ //

export function isDiscoveryOutput(
  output: RunMCPToolOutput,
): output is MCPToolsDiscoveredOutput {
  return output.type === MCP_TOOLS_DISCOVERED;
}

export function isMCPToolOutput(
  output: RunMCPToolOutput,
): output is MCPToolOutputResult {
  return output.type === MCP_TOOL_OUTPUT;
}

export function isSetupRequirementsOutput(
  output: RunMCPToolOutput,
): output is SetupRequirementsResponse {
  return (
    output.type === SETUP_REQUIREMENTS ||
    ("setup_info" in output && typeof output.setup_info === "object")
  );
}

export function isErrorOutput(
  output: RunMCPToolOutput,
): output is MCPErrorOutput {
  return (
    output.type === ERROR || ("error" in output && !("setup_info" in output))
  );
}

// ------------------------------------------------------------------ //
//  Output parsing
// ------------------------------------------------------------------ //

function parseOutput(raw: unknown): RunMCPToolOutput | null {
  if (!raw) return null;
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (!trimmed) return null;
    try {
      return parseOutput(JSON.parse(trimmed) as unknown);
    } catch {
      return null;
    }
  }
  if (typeof raw === "object") {
    const type = (raw as { type?: unknown }).type;
    if (typeof type === "string" && RUN_MCP_TOOL_OUTPUT_TYPES.has(type)) {
      return raw as RunMCPToolOutput;
    }
    // Fallback structural checks
    if ("setup_info" in (raw as object))
      return raw as SetupRequirementsResponse;
    if ("tool_name" in (raw as object)) return raw as MCPToolOutputResult;
    if ("tools" in (raw as object)) return raw as MCPToolsDiscoveredOutput;
  }
  return null;
}

export function getRunMCPToolOutput(part: unknown): RunMCPToolOutput | null {
  if (!part || typeof part !== "object") return null;
  return parseOutput((part as { output?: unknown }).output);
}

// ------------------------------------------------------------------ //
//  UI helpers
// ------------------------------------------------------------------ //

export interface RunMCPToolInput {
  server_url?: string;
  tool_name?: string;
}

export function serverHost(url: string): string {
  try {
    return new URL(url).hostname || url;
  } catch {
    return url;
  }
}

export function getAnimationText(part: {
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}): string {
  const input = part.input as RunMCPToolInput | undefined;
  const host = input?.server_url ? serverHost(input.server_url) : "";
  const toolName = input?.tool_name?.trim();

  switch (part.state) {
    case "input-streaming":
    case "input-available":
      return toolName
        ? `Calling ${toolName}${host ? ` on ${host}` : ""}`
        : `Discovering MCP tools${host ? ` on ${host}` : ""}`;
    case "output-available": {
      const output = getRunMCPToolOutput(part);
      if (!output) return "Connecting to MCP server";
      if (isSetupRequirementsOutput(output))
        return `Connect to ${output.setup_info.agent_name}`;
      if (isMCPToolOutput(output))
        return `Ran ${output.tool_name}${host ? ` on ${host}` : ""}`;
      if (isDiscoveryOutput(output))
        return `Discovered ${output.tools.length} tool(s) on ${serverHost(output.server_url)}`;
      return "MCP error";
    }
    case "output-error":
      return "MCP error";
    default:
      return "Connecting to MCP server";
  }
}

export function ToolIcon({
  isStreaming,
  isError,
}: {
  isStreaming?: boolean;
  isError?: boolean;
}) {
  if (isError) {
    return (
      <WarningDiamondIcon size={14} weight="regular" className="text-red-500" />
    );
  }
  if (isStreaming) {
    return <OrbitLoader size={24} />;
  }
  return (
    <PlugsConnectedIcon
      size={14}
      weight="regular"
      className="text-neutral-400"
    />
  );
}
