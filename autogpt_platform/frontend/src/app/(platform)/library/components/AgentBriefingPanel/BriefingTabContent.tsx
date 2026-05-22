"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { useSitrepItems } from "../SitrepItem/useSitrepItems";
import { SitrepItem } from "../SitrepItem/SitrepItem";
import { useAgentStatusMap } from "../../hooks/useAgentStatus";
import type { AgentStatusFilter } from "../../types";
import { Text } from "@/components/atoms/Text/Text";
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
