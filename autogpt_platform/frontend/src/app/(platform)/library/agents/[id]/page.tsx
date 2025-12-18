"use client";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { NewAgentLibraryView } from "./components/NewAgentLibraryView/NewAgentLibraryView";
import { OldAgentLibraryView } from "./components/OldAgentLibraryView/OldAgentLibraryView";

export default function AgentLibraryPage() {
  const isNewLibraryPageEnabled = useGetFlag(Flag.NEW_AGENT_RUNS);
  return isNewLibraryPageEnabled ? (
    <NewAgentLibraryView />
  ) : (
    <OldAgentLibraryView />
  );
}
