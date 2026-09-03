"use client";

import { useSitrepItems } from "@/app/(platform)/library/components/SitrepItem/useSitrepItems";
import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";

const MAX_RUNS = 5;

/** Each workflow's latest state from the library feed: a recent result, a
 *  live run, or a trigger it waits on. Scheduled runs are left out because
 *  Now & next already lists them under Coming up. */
export function useWorkflowRuns() {
  const { agents } = useLibraryAgents();
  return useSitrepItems(agents, MAX_RUNS * 2)
    .filter((run) => run.status !== "scheduled")
    .slice(0, MAX_RUNS);
}
