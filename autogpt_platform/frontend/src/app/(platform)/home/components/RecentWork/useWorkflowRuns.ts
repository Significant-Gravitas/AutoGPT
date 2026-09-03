"use client";

import { useSitrepItems } from "@/app/(platform)/library/components/SitrepItem/useSitrepItems";
import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";

const MAX_RUNS = 5;
const THREE_DAYS_MS = 3 * 24 * 60 * 60 * 1000;

/** Each workflow's latest state from the library feed: a recent result,
 *  a live run, a trigger it waits on, or a run scheduled within three days. */
export function useWorkflowRuns() {
  const { agents } = useLibraryAgents();
  return useSitrepItems(agents, MAX_RUNS, THREE_DAYS_MS);
}
