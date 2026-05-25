"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useListCopilotSkills } from "@/app/api/__generated__/endpoints/skills/skills";
import { useListCopilotFollowupSchedules } from "@/app/api/__generated__/endpoints/schedules/schedules";
import { Button } from "@/components/atoms/Button/Button";
import { useSitrepItems } from "../SitrepItem/useSitrepItems";
import { SitrepItem } from "../SitrepItem/SitrepItem";
import { useAgentStatusMap } from "../../hooks/useAgentStatus";
import type { AgentStatusFilter } from "../../types";
import { Text } from "@/components/atoms/Text/Text";
import Link from "next/link";
import { useState } from "react";
import { CostsBreakdown } from "./CostsBreakdown/CostsBreakdown";

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
    return <ExecutionListSection activeTab={activeTab} agents={agents} />;
  }

  return <AgentListSection activeTab={activeTab} agents={agents} />;
}

function CopilotLibrarySummary() {
  // Discoverability is already gated by AGENT_BRIEFING at the parent
  // panel — this pill renders only inside AgentBriefingPanel, which is
  // itself flag-gated.  No second flag here because the count-based
  // hide below already keeps the pill quiet for users who don't use
  // the feature.
  const { data: skillsRes } = useListCopilotSkills({
    query: { staleTime: 30_000 },
  });
  const { data: followupsRes } = useListCopilotFollowupSchedules({
    query: { staleTime: 30_000 },
  });

  const skillsCount =
    skillsRes && skillsRes.status === 200 ? skillsRes.data.length : 0;
  // Count only copilot follow-ups here — graph schedules (recurring
  // agent runs) are already surfaced by the briefing's own "Scheduled"
  // tab above, so folding them into this pill would double-count and
  // confuse the "Autopilot library" framing.  The pill's link still
  // goes to the unified `/library/followups` page, where both kinds
  // are listed together.
  const followupCount =
    followupsRes && followupsRes.status === 200 ? followupsRes.data.length : 0;

  // Suppress the pill entirely when the user has no autopilot library
  // content yet — surfacing "0 skills · 0 follow-ups" is noise, not a
  // discovery affordance.  The pill reappears the moment either count
  // turns positive (e.g. after a store_skill / schedule_followup tool
  // call).
  if (skillsCount === 0 && followupCount === 0) {
    return null;
  }

  // Per-link hide: surface only the counts that are non-zero.  We
  // already returned ``null`` above when both are zero, so at least
  // one branch always renders.
  const showSkills = skillsCount > 0;
  const showFollowups = followupCount > 0;

  return (
    <div
      className="mt-5 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-zinc-100 pt-3"
      data-testid="copilot-library-summary"
    >
      <Text variant="small" className="!text-zinc-500">
        Autopilot library
      </Text>
      {showSkills ? (
        <Link
          href="/library/skills"
          className="text-sm text-violet-700 hover:underline"
          data-testid="copilot-library-skills-link"
        >
          {skillsCount} skill{skillsCount === 1 ? "" : "s"}
        </Link>
      ) : null}
      {showSkills && showFollowups ? (
        <span className="text-zinc-300">•</span>
      ) : null}
      {showFollowups ? (
        <Link
          href="/library/followups"
          className="text-sm text-yellow-700 hover:underline"
          data-testid="copilot-library-followups-link"
        >
          {followupCount} follow-up{followupCount === 1 ? "" : "s"}
        </Link>
      ) : null}
    </div>
  );
}

const MAX_VISIBLE = 6;

function ExecutionListSection({
  activeTab,
  agents,
}: {
  activeTab: AgentStatusFilter;
  agents: LibraryAgent[];
}) {
  const allItems = useSitrepItems(agents, 50);
  const [showAll, setShowAll] = useState(false);

  const filtered = allItems.filter((item) => {
    if (activeTab === "running") return item.priority === "running";
    if (activeTab === "attention") return item.priority === "error";
    if (activeTab === "completed") return item.priority === "success";
    return false;
  });

  if (filtered.length === 0) {
    return <EmptyMessage tab={activeTab} />;
  }

  const visible = showAll ? filtered : filtered.slice(0, MAX_VISIBLE);
  const hasMore = filtered.length > MAX_VISIBLE;

  return (
    <div>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {visible.map((item) => (
          <SitrepItem key={item.id} item={item} />
        ))}
      </div>
      {hasMore && (
        <div className="mt-3 flex justify-center">
          <Button
            variant="secondary"
            size="small"
            onClick={() => setShowAll(!showAll)}
          >
            {showAll ? "Collapse" : `Show all (${filtered.length})`}
          </Button>
        </div>
      )}
    </div>
  );
}

const TAB_STATUS_LABEL: Record<string, string> = {
  listening: "Waiting for trigger event",
  scheduled: "Has a scheduled run",
  idle: "No recent activity",
};

function getAgentStatusLabel(tab: string, agent: LibraryAgent): string {
  if (tab === "scheduled" && agent.next_scheduled_run) {
    const diff = new Date(agent.next_scheduled_run).getTime() - Date.now();
    const minutes = Math.round(diff / 60_000);
    if (minutes <= 0) return "Scheduled to run soon";
    if (minutes < 60) return `Scheduled to run in ${minutes}m`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `Scheduled to run in ${hours}h`;
    const days = Math.round(hours / 24);
    return `Scheduled to run in ${days}d`;
  }
  return TAB_STATUS_LABEL[tab] ?? "";
}

function AgentListSection({
  activeTab,
  agents,
}: {
  activeTab: AgentStatusFilter;
  agents: LibraryAgent[];
}) {
  const [showAll, setShowAll] = useState(false);
  const statusMap = useAgentStatusMap(agents);

  const filtered = agents.filter((agent) => {
    const status = statusMap.get(agent.graph_id)?.status;
    if (activeTab === "listening") return status === "listening";
    if (activeTab === "scheduled") return status === "scheduled";
    if (activeTab === "idle") return status === "idle";
    return false;
  });

  if (filtered.length === 0) {
    return <EmptyMessage tab={activeTab} />;
  }

  const status =
    activeTab === "listening"
      ? ("listening" as const)
      : activeTab === "scheduled"
        ? ("scheduled" as const)
        : ("idle" as const);

  const visible = showAll ? filtered : filtered.slice(0, MAX_VISIBLE);
  const hasMore = filtered.length > MAX_VISIBLE;

  return (
    <div>
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {visible.map((agent) => (
          <SitrepItem
            key={agent.id}
            item={{
              id: agent.id,
              agentID: agent.id,
              agentName: agent.name,
              agentImageUrl: agent.image_url,
              priority: status,
              message: getAgentStatusLabel(activeTab, agent),
              status,
            }}
          />
        ))}
      </div>
      {hasMore && (
        <div className="mt-3 flex justify-center">
          <Button
            variant="secondary"
            size="small"
            onClick={() => setShowAll(!showAll)}
          >
            {showAll ? "Collapse" : `Show all (${filtered.length})`}
          </Button>
        </div>
      )}
    </div>
  );
}

const EMPTY_MESSAGES: Record<string, string> = {
  running: "No agents running right now",
  attention: "No agents that need attention",
  completed: "No recently completed runs",
  listening: "No agents listening for events",
  scheduled: "No agents with scheduled runs",
  idle: "No idle agents",
};

function EmptyMessage({ tab }: { tab: AgentStatusFilter }) {
  return (
    <div className="flex items-center justify-center pt-4">
      <Text variant="body-medium" className="text-zinc-600">
        {EMPTY_MESSAGES[tab] ?? "No agents in this category"}
      </Text>
    </div>
  );
}
