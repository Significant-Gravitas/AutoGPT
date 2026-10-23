import { AGENT_TOOL_CATALOG } from "./toolCatalog.agent";
import { PLATFORM_TOOL_CATALOG } from "./toolCatalog.platform";
import type {
  ChainCategory,
  ToolDisplayContext,
  ToolInput,
  ToolMeta,
} from "./toolCatalog.shared";

export type { ChainCategory } from "./toolCatalog.shared";

export const COPILOT_TOOL_CATALOG: Record<string, ToolMeta> = {
  ...AGENT_TOOL_CATALOG,
  ...PLATFORM_TOOL_CATALOG,
};

export function getCatalogLabel(
  toolName: string,
  input: unknown,
  state: "running" | "done" | "error",
  context: ToolDisplayContext = {},
): { category: ChainCategory; text: string } | null {
  const meta = COPILOT_TOOL_CATALOG[toolName];
  if (!meta) return null;
  const fields = input && typeof input === "object" ? (input as ToolInput) : {};
  const phrase = meta.phrase?.(fields) ?? null;
  if (phrase)
    return { category: meta.category, text: phraseLabel(phrase, state) };
  const subject = meta.subject?.(fields, context) ?? null;
  const suffix = subject ? ` ${subject}` : "";
  const text =
    state === "running"
      ? `${meta.running}${suffix}…`
      : state === "error"
        ? `Failed while ${meta.running.charAt(0).toLowerCase()}${meta.running.slice(1)}${suffix}`
        : `${meta.done}${suffix}`;
  return { category: meta.category, text };
}

function phraseLabel(phrase: string, state: "running" | "done" | "error") {
  // A phrase cut to fit already ends in one.
  if (state === "running") return phrase.endsWith("…") ? phrase : `${phrase}…`;
  if (state === "done") return phrase;
  // "Updating your brief" reads on after "Failed while"; a noun phrase
  // ("Shopify sync") does not, so it follows a colon with its capital kept.
  const [first] = phrase.split(" ");
  return /^[A-Z][a-z]*ing$/.test(first)
    ? `Failed while ${phrase.charAt(0).toLowerCase()}${phrase.slice(1)}`
    : `Failed: ${phrase}`;
}
