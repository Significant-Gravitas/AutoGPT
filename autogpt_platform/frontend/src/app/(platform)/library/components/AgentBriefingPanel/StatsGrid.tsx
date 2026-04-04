"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  CurrencyDollarIcon,
  PlayCircleIcon,
  WarningCircleIcon,
  EarIcon,
  ClockIcon,
  PauseCircleIcon,
} from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import type { FleetSummary, AgentStatusFilter } from "../../types";

interface Props {
  summary: FleetSummary;
  activeFilter: AgentStatusFilter;
  onFilterChange?: (filter: AgentStatusFilter) => void;
}

export function StatsGrid({ summary, activeFilter, onFilterChange }: Props) {
  const tiles = [
    {
      label: "Spend this month",
      value: `$${summary.monthlySpend.toLocaleString()}`,
      filter: "all" as AgentStatusFilter,
      icon: CurrencyDollarIcon,
      color: "text-zinc-700",
    },
    {
      label: "Running now",
      value: summary.running,
      filter: "running" as AgentStatusFilter,
      icon: PlayCircleIcon,
      color: "text-blue-600",
    },
    {
      label: "Needs attention",
      value: summary.error,
      filter: "attention" as AgentStatusFilter,
      icon: WarningCircleIcon,
      color: "text-red-500",
    },
    {
      label: "Listening",
      value: summary.listening,
      filter: "listening" as AgentStatusFilter,
      icon: EarIcon,
      color: "text-purple-500",
    },
    {
      label: "Scheduled",
      value: summary.scheduled,
      filter: "scheduled" as AgentStatusFilter,
      icon: ClockIcon,
      color: "text-yellow-600",
    },
    {
      label: "Idle",
      value: summary.idle,
      filter: "idle" as AgentStatusFilter,
      icon: PauseCircleIcon,
      color: "text-zinc-400",
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
      {tiles.map((tile) => {
        const Icon = tile.icon;
        const isActive = activeFilter === tile.filter;

        return (
          <button
            key={tile.label}
            type="button"
            onClick={() => onFilterChange?.(tile.filter)}
            className={cn(
              "flex flex-col gap-1 rounded-medium border p-3 text-left transition-all hover:shadow-sm",
              isActive
                ? "border-zinc-900 bg-zinc-50"
                : "border-zinc-100 bg-white",
            )}
          >
            <div className="flex items-center gap-1.5">
              <Icon size={14} className={tile.color} />
              <Text variant="small" className="text-zinc-500">
                {tile.label}
              </Text>
            </div>
            <Text variant="h4" className={tile.color}>
              {tile.value}
            </Text>
          </button>
        );
      })}
    </div>
  );
}
