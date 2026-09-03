"use client";

import { useSitrepItems } from "@/app/(platform)/library/components/SitrepItem/useSitrepItems";
import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";

const MAX_RUNS = 5;
const THREE_DAYS_MS = 3 * 24 * 60 * 60 * 1000;

export function useRecentWorkflowRuns() {
  const { agents } = useLibraryAgents();
  return useSitrepItems(agents, MAX_RUNS, THREE_DAYS_MS);
}
