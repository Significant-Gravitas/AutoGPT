"use client";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { AgentRunsView } from "./components/AgentRunsView/AgentRunsView";
import { OldAgentLibraryView } from "./components/OldAgentLibraryView/OldAgentLibraryView";

export default function AgentLibraryPage() {
  const isNewLibraryPageEnabled = useGetFlag(Flag.NEW_AGENT_RUNS);
  return isNewLibraryPageEnabled ? <AgentRunsView /> : <OldAgentLibraryView />;
}
