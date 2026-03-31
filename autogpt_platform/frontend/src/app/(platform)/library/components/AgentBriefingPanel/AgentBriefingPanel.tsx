"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { CaretUpIcon, CaretDownIcon } from "@phosphor-icons/react";
import { useState } from "react";
import type { FleetSummary, AgentStatusFilter } from "../../types";
import { SitrepList } from "../SitrepItem/SitrepList";
import { StatsGrid } from "./StatsGrid";

interface Props {
  summary: FleetSummary;
  agentIDs: string[];
  onFilterChange?: (filter: AgentStatusFilter) => void;
  activeFilter?: AgentStatusFilter;
}

export function AgentBriefingPanel({
  summary,
  agentIDs,
  onFilterChange,
  activeFilter = "all",
}: Props) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  const totalAttention = summary.error;

  const headerSummary = [
    summary.running > 0 && `${summary.running} running`,
    totalAttention > 0 && `${totalAttention} need attention`,
    summary.listening > 0 && `${summary.listening} listening`,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="rounded-large border border-zinc-100 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Text variant="h5">Agent Briefing</Text>
          {headerSummary && (
            <Text variant="small" className="text-zinc-500">
              {headerSummary}
            </Text>
          )}
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsCollapsed(!isCollapsed)}
          aria-label={isCollapsed ? "Expand briefing" : "Collapse briefing"}
        >
          {isCollapsed ? (
            <CaretDownIcon size={16} />
          ) : (
            <CaretUpIcon size={16} />
          )}
        </Button>
      </div>

      {!isCollapsed && (
        <div className="mt-4 space-y-5">
          <StatsGrid
            summary={summary}
            activeFilter={activeFilter}
            onFilterChange={onFilterChange}
          />
          <SitrepList agentIDs={agentIDs} />
        </div>
      )}
    </div>
  );
}
