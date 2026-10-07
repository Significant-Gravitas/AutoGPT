"use client";

import { Text } from "@/components/atoms/Text/Text";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useState } from "react";
import type { FleetSummary, AgentStatusFilter } from "../../types";
import { BriefingTabContent } from "./components/BriefingTabContent/BriefingTabContent";
import { StatsGrid } from "./components/StatsGrid/StatsGrid";

interface Props {
  summary: FleetSummary;
  agents: LibraryAgent[];
}

export function AgentBriefingPanel({ summary, agents }: Props) {
  const [userTab, setUserTab] = useState<AgentStatusFilter | null>(null);
  const activeTab: AgentStatusFilter =
    userTab ?? (summary.running > 0 ? "running" : "all");

  return (
    <div
      className={`min-h-59 rounded-large bg-linear-to-br from-purple-50/30 via-white/90 to-purple-50/25 px-5 pt-4.5 pb-5 shadow-xs backdrop-blur-md`}
    >
      <Text variant="h5">Agent Briefing</Text>
      <div className="mt-4 space-y-5">
        <StatsGrid
          summary={summary}
          activeTab={activeTab}
          onTabChange={setUserTab}
        />
        <BriefingTabContent activeTab={activeTab} agents={agents} />
      </div>
    </div>
  );
}
