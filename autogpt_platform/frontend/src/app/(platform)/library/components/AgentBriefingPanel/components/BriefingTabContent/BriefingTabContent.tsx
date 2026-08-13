"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";

import { CostsBreakdown } from "../CostsBreakdown/CostsBreakdown";
import { AgentListSection } from "./components/AgentListSection";
import { CopilotLibrarySummary } from "./components/CopilotLibrarySummary";
import { ExecutionListSection } from "./components/ExecutionListSection";
import { AgentStatusFilter } from "@/app/(platform)/library/types";

interface Props {
  activeTab: AgentStatusFilter;
  agents: LibraryAgent[];
}

export function BriefingTabContent({ activeTab, agents }: Props) {
  if (activeTab === "all") {
    return (
      <div className="py-2">
        <CostsBreakdown agents={agents} />
        <CopilotLibrarySummary />
      </div>
    );
  }

  if (
    activeTab === "running" ||
    activeTab === "attention" ||
    activeTab === "completed"
  ) {
    return (
      <ExecutionListSection
        key={activeTab}
        activeTab={activeTab}
        agents={agents}
      />
    );
  }

  return (
    <AgentListSection key={activeTab} activeTab={activeTab} agents={agents} />
  );
}
