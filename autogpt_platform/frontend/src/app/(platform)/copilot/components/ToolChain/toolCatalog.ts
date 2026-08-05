import { AGENT_TOOL_CATALOG } from "./toolCatalog.agent";
import { PLATFORM_TOOL_CATALOG } from "./toolCatalog.platform";
import type { ChainCategory, ToolInput, ToolMeta } from "./toolCatalog.shared";

export type { ChainCategory } from "./toolCatalog.shared";

export const COPILOT_TOOL_CATALOG: Record<string, ToolMeta> = {
  ...AGENT_TOOL_CATALOG,
  ...PLATFORM_TOOL_CATALOG,
};

export function getCatalogLabel(
  toolName: string,
  input: unknown,
  state: "running" | "done" | "error",
): { category: ChainCategory; text: string } | null {
  const meta = COPILOT_TOOL_CATALOG[toolName];
  if (!meta) return null;
  const subject =
    meta.subject?.(
      input && typeof input === "object" ? (input as ToolInput) : {},
    ) ?? null;
  const suffix = subject ? ` ${subject}` : "";
  const text =
    state === "running"
      ? `${meta.running}${suffix}…`
      : state === "error"
        ? `Failed while ${meta.running.charAt(0).toLowerCase()}${meta.running.slice(1)}${suffix}`
        : `${meta.done}${suffix}`;
  return { category: meta.category, text };
}
