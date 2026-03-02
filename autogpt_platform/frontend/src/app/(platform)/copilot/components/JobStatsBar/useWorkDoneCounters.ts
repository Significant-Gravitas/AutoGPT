import { useMemo } from "react";
import type { UIDataTypes, UIMessage, UITools } from "ai";

/**
 * Counter categories that map tool names to human-readable labels.
 * Only "meaningful" external actions are counted -- internal operations
 * (like add_understanding, search_docs, get_doc_page) are excluded.
 */
const TOOL_TO_CATEGORY: Record<string, string> = {
  // Searches
  find_agent: "searches",
  find_library_agent: "searches",

  // Agent runs
  run_agent: "agents run",
  run_block: "blocks run",

  // Agent creation / editing
  create_agent: "agents created",
  edit_agent: "agents edited",

  // Scheduling
  schedule_agent: "agents scheduled",
};

/** Maximum number of counter categories to display */
const MAX_COUNTERS = 3;

interface WorkDoneCounter {
  label: string;
  count: number;
}

export function useWorkDoneCounters(
  messages: UIMessage<unknown, UIDataTypes, UITools>[],
) {
  const counters = useMemo(
    function computeCounters(): WorkDoneCounter[] {
      const counts = new Map<string, number>();

      for (const message of messages) {
        if (message.role !== "assistant") continue;

        for (const part of message.parts) {
          // Tool parts have types like "tool-run_agent", "tool-find_agent"
          if (!part.type.startsWith("tool-")) continue;

          const toolName = part.type.replace("tool-", "");
          const category = TOOL_TO_CATEGORY[toolName];
          if (!category) continue;

          counts.set(category, (counts.get(category) ?? 0) + 1);
        }
      }

      // Sort by count descending, then take the top N
      const sorted = Array.from(counts.entries())
        .map(function toCounter([label, count]) {
          return { label, count };
        })
        .sort(function byCountDesc(a, b) {
          return b.count - a.count;
        })
        .slice(0, MAX_COUNTERS);

      return sorted;
    },
    [messages],
  );

  return { counters };
}
