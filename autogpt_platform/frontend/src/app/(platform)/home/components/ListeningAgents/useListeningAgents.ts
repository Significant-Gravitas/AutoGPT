"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useLibraryAgents } from "@/hooks/useLibraryAgents/useLibraryAgents";

/** Agents waiting on an external trigger. Every other run state already has
 *  a home on the page: finished runs are outcomes, live ones are "Working
 *  now", scheduled ones are "Coming up". */
export function useListeningAgents(): LibraryAgent[] {
  const { agents } = useLibraryAgents();
  return agents.filter((agent) => agent.has_external_trigger);
}
