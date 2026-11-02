"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { useState } from "react";
import { getAgentStatusLabel, MAX_VISIBLE } from "../helpers";
import { EmptyMessage } from "./EmptyMessage";
import { AgentStatusFilter } from "@/app/(platform)/library/types";
import { useAgentStatusMap } from "@/app/(platform)/library/hooks/useAgentStatus";
import { SitrepItem } from "@/app/(platform)/library/components/SitrepItem/SitrepItem";

interface Props {
  activeTab: AgentStatusFilter;
  agents: LibraryAgent[];
}

export function AgentListSection({ activeTab, agents }: Props) {
  const [showAll, setShowAll] = useState(false);
  const statusMap = useAgentStatusMap(agents);

  const filtered = agents.filter((agent) => {
    const status = statusMap.get(agent.graph_id)?.status;
    if (activeTab === "listening") return status === "listening";
    if (activeTab === "scheduled") return status === "scheduled";
    if (activeTab === "idle") return status === "idle";
    return false;
  });

  if (filtered.length === 0) return <EmptyMessage tab={activeTab} />;

  const status =
    activeTab === "listening"
      ? ("listening" as const)
      : activeTab === "scheduled"
        ? ("scheduled" as const)
        : ("idle" as const);

  const visible = showAll ? filtered : filtered.slice(0, MAX_VISIBLE);
  const hasMore = filtered.length > MAX_VISIBLE;

  function handleToggleShowAll() {
    setShowAll(!showAll);
  }

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
      {hasMore ? (
        <div className="mt-3 flex justify-center">
          <Button
            variant="secondary"
            size="small"
            onClick={handleToggleShowAll}
          >
            {showAll ? "Collapse" : `Show all (${filtered.length})`}
          </Button>
        </div>
      ) : null}
    </div>
  );
}
