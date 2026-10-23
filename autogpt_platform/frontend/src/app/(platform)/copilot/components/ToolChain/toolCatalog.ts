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
  if (state === "running") return `${phrase}…`;
  if (state === "done") return phrase;
  // "Updating your brief" reads on after "Failed while"; a phrase that opens
  // with a name ("Shopify sync") keeps its capital.
  const [first] = phrase.split(" ");
  const lead = /^[A-Z][a-z]*ing$/.test(first)
    ? phrase.charAt(0).toLowerCase() + phrase.slice(1)
    : phrase;
  return `Failed while ${lead}`;
}
