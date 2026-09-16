"use client";

import { useSitrepItems } from "@/app/(platform)/library/components/SitrepItem/useSitrepItems";
import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";

const MAX_RUNS = 5;

/** Each workflow's latest state from the library feed: a recent result, a
 *  live run, or a trigger it waits on. Scheduled runs are left out because
 *  Now & next already lists them under Coming up. */
export function useWorkflowRuns() {
  const { agents } = useLibraryAgents();
  // The feed yields at most one item per agent, so asking for that many
  // makes its own cap a no-op: scheduled rows are dropped before the real
  // limit applies, and a run of scheduled agents cannot crowd out the rest.
  return useSitrepItems(agents, Math.max(agents.length, 1))
    .filter((run) => run.status !== "scheduled")
    .slice(0, MAX_RUNS);
}
