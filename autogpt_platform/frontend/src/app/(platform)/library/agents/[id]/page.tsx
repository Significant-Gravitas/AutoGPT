"use client";

import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

import { OldAgentLibraryView } from "./components/OldAgentLibraryView/OldAgentLibraryView";
import { AgentRunsView } from "./components/AgentRunsView/AgentRunsView";

export default function AgentLibraryPage() {
  const isNewAgentRunsEnabled = useGetFlag(Flag.NEW_AGENT_RUNS);

  if (isNewAgentRunsEnabled) {
    return <AgentRunsView />;
  }

  return <OldAgentLibraryView />;
}
