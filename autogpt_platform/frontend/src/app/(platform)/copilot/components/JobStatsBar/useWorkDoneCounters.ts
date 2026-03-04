import type { UIDataTypes, UIMessage, UITools } from "ai";

/**
 * Counter categories that map tool names to singular human-readable labels.
 * Only "meaningful" external actions are counted -- internal operations
 * (like add_understanding, search_docs, get_doc_page) are excluded.
 */
const TOOL_TO_CATEGORY: Record<string, string> = {
  // Searches
  find_agent: "search",
  find_library_agent: "search",

  // Agent runs
  run_agent: "agent run",
  run_block: "block run",

  // Agent creation / editing
  create_agent: "agent created",
  edit_agent: "agent edited",

  // Scheduling
  schedule_agent: "agent scheduled",
};

/** Maximum number of counter categories to display */
const MAX_COUNTERS = 3;

function pluralize(label: string, count: number): string {
  if (count === 1) return label;

  // "agent created" -> "agents created", "agent edited" -> "agents edited"
  const nounVerbMatch = label.match(
    /^(\w+)\s+(created|edited|scheduled|run)$/i,
  );
  if (nounVerbMatch) {
    return pluralizeWord(nounVerbMatch[1]) + " " + nounVerbMatch[2];
  }

  return pluralizeWord(label);
}

function pluralizeWord(word: string): string {
  if (word.endsWith("ch") || word.endsWith("sh") || word.endsWith("x"))
    return word + "es";
  return word + "s";
}

export interface WorkDoneCounter {
  label: string;
  count: number;
  category: string;
}

export function useWorkDoneCounters(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
) {
  const categoryCounts = new Map<string, number>();

  for (const message of messages) {
    if (message.role !== "assistant") continue;

    for (const part of message.parts) {
      if (!part.type.startsWith("tool-")) continue;

      const toolName = part.type.replace("tool-", "");
      const category = TOOL_TO_CATEGORY[toolName];
      if (!category) continue;

      categoryCounts.set(category, (categoryCounts.get(category) ?? 0) + 1);
    }
  }

  const counters: WorkDoneCounter[] = Array.from(categoryCounts.entries())
    .map(function toCounter([category, count]) {
      return {
        label: pluralize(category, count),
        count,
        category,
      };
    })
    .sort(function byCountDesc(a, b) {
      return b.count - a.count;
    })
    .slice(0, MAX_COUNTERS);

  return { counters };
}
