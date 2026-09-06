import type { ToolUIPart, UIMessage } from "ai";
import { beautifyString } from "@/lib/utils";
import { asObject } from "../components/ToolChain/resultHelpers";

export function toolDisplayName(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

export function withToolDisplayNames(
  parts: UIMessage["parts"],
): UIMessage["parts"] {
  const names = new Map<string, string>();
  for (const part of parts) {
    if (part.type !== "data-tool-display" || !("data" in part)) continue;
    const data = asObject(part.data);
    const toolCallID = toolDisplayName(data?.toolCallId);
    const name = toolDisplayName(data?.displayName);
    if (toolCallID && name) names.set(toolCallID, name);
  }
  if (names.size === 0) return parts;
  return parts.map((part) => {
    if (!part.type.startsWith("tool-")) return part;
    const tool = part as ToolUIPart;
    const title = names.get(tool.toolCallId);
    return title ? { ...tool, title } : part;
  });
}

export function getAgentDisplayName(
  displayName: unknown,
  output?: unknown,
): string | null {
  const name = toolDisplayName(displayName);
  if (name) return name;
  const result = asObject(output);
  if (!result) return null;
  const resultName =
    toolDisplayName(result.graph_name) ?? toolDisplayName(result.agent_name);
  if (resultName) return resultName;
  if (result.type === "agent_details") {
    return toolDisplayName(asObject(result.agent)?.name);
  }
  if (result.type === "setup_requirements") {
    return toolDisplayName(asObject(result.setup_info)?.agent_name);
  }
  return null;
}

export function getBlockDisplayName(
  displayName: unknown,
  output?: unknown,
): string | null {
  const result = asObject(output);
  const name =
    toolDisplayName(displayName) ??
    toolDisplayName(result?.block_name) ??
    toolDisplayName(asObject(result?.block)?.name) ??
    (result?.type === "setup_requirements"
      ? toolDisplayName(asObject(result.setup_info)?.agent_name)
      : null);
  return name
    ? beautifyString(name)
        .replace(/ Block$/, "")
        .trim()
    : null;
}
