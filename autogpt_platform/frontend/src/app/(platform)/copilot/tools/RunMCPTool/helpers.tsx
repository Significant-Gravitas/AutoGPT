"use client";

import type { MCPToolOutputResponse } from "@/app/api/__generated__/models/mCPToolOutputResponse";
import type { MCPToolsDiscoveredResponse } from "@/app/api/__generated__/models/mCPToolsDiscoveredResponse";
import { ResponseType } from "@/app/api/__generated__/models/responseType";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { WarningDiamondIcon, PlugsConnectedIcon } from "@phosphor-icons/react";
import type { ToolUIPart } from "ai";
import { OrbitLoader } from "../../components/OrbitLoader/OrbitLoader";

// ------------------------------------------------------------------ //
//  Re-export generated types for use by RunMCPTool components
// ------------------------------------------------------------------ //

export type { MCPToolsDiscoveredResponse, MCPToolOutputResponse };

export interface MCPErrorOutput {
  type: typeof ResponseType.error;
  message: string;
  error?: string | null;
  session_id?: string | null;
}

export type RunMCPToolOutput =
  | MCPToolsDiscoveredResponse
  | MCPToolOutputResponse
  | SetupRequirementsResponse
  | MCPErrorOutput;

// ------------------------------------------------------------------ //
//  Type guards
// ------------------------------------------------------------------ //

export function isDiscoveryOutput(
  output: RunMCPToolOutput,
): output is MCPToolsDiscoveredResponse {
  return output.type === ResponseType.mcp_tools_discovered;
}

export function isMCPToolOutput(
  output: RunMCPToolOutput,
): output is MCPToolOutputResponse {
  return output.type === ResponseType.mcp_tool_output;
}

export function isSetupRequirementsOutput(
  output: RunMCPToolOutput,
): output is SetupRequirementsResponse {
  return (
    output.type === ResponseType.setup_requirements ||
    ("setup_info" in output && typeof output.setup_info === "object")
  );
}

/** Returns true only when the response type is explicitly "error". */
export function isErrorOutput(
  output: RunMCPToolOutput,
): output is MCPErrorOutput {
  return output.type === ResponseType.error;
}

// ------------------------------------------------------------------ //
//  Output parsing
// ------------------------------------------------------------------ //

/**
 * Parse a raw server payload into a typed RunMCPToolOutput.
 * Accepts both objects (already parsed) and JSON strings.
 * Returns null for anything that doesn't look like a known response shape.
 */
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
    if (
      typeof type === "string" &&
      (Object.values(ResponseType) as string[]).includes(type)
    ) {
      return raw as RunMCPToolOutput;
    }
    // Fallback structural checks for legacy / no-type payloads
    if ("setup_info" in (raw as object))
      return raw as SetupRequirementsResponse;
    if ("tool_name" in (raw as object)) return raw as MCPToolOutputResponse;
    if ("tools" in (raw as object)) return raw as MCPToolsDiscoveredResponse;
  }
  return null;
}

/**
 * Extract and parse the `output` field from a tool UI part.
 * Returns null when the output is absent or unrecognised.
 */
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
  tool_arguments?: Record<string, unknown>;
}

export function serverHost(url: string): string {
  try {
    return new URL(url).hostname || url;
  } catch {
    return url;
  }
}

/**
 * Return a short preview of the most meaningful argument value, e.g. `"my query"`.
 * Checks common "query" key names first, then falls back to the first string value.
 * Returns an empty string when no suitable argument is found.
 */
function getArgPreview(args: Record<string, unknown> | undefined): string {
  if (!args) return "";
  const queryKeys = [
    "query",
    "q",
    "search",
    "text",
    "content",
    "message",
    "input",
    "prompt",
  ];
  for (const key of queryKeys) {
    if (typeof args[key] === "string" && (args[key] as string).length > 0)
      return `"${args[key]}"`;
  }
  for (const val of Object.values(args)) {
    if (typeof val === "string" && val.length > 0) return `"${val}"`;
  }
  return "";
}

/**
 * Return the human-readable status line shown next to the MCP tool spinner.
 * Transitions through: connecting → discovering / calling → result summary.
 */
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
    case "input-available": {
      if (!toolName) return `Discovering MCP tools${host ? ` on ${host}` : ""}`;
      const argPreview = getArgPreview(input?.tool_arguments);
      return `Calling ${toolName}${argPreview ? `(${argPreview})` : ""}${host ? ` on ${host}` : ""}`;
    }
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
