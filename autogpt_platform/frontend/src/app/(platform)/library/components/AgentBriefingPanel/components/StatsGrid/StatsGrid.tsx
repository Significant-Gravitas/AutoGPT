"use client";

import { Text } from "@/components/atoms/Text/Text";
import { OverflowText } from "@/components/atoms/OverflowText/OverflowText";
import { Emoji } from "@/components/atoms/Emoji/Emoji";
import { cn } from "@/lib/utils";
import { formatCents } from "@/app/(platform)/copilot/components/usageHelpers";
import { useIsBillingEnabled } from "@/hooks/useIsBillingEnabled";
import type { FleetSummary, AgentStatusFilter } from "../../../../types";

interface Props {
  summary: FleetSummary;
  activeTab: AgentStatusFilter;
  onTabChange: (tab: AgentStatusFilter) => void;
}

const TILES: {
  label: string;
  key: Exclude<keyof FleetSummary, "schedulesUnknown">;
  format?: (v: number) => string;
  filter: AgentStatusFilter;
  emoji: string;
  color: string;
}[] = [
  {
    label: "Spent this month",
    key: "monthlySpend",
    format: formatCents,
    filter: "all",
    emoji: "💵",
    color: "text-foreground-body",
  },
  {
    label: "Running now",
    key: "running",
    filter: "running",
    emoji: "🚩",
    color: "text-info-foreground",
  },
  {
    label: "Recently completed",
    key: "completed",
    filter: "completed",
    emoji: "🗃️",
    color: "text-success",
  },
  {
    label: "Needs attention",
    key: "error",
    filter: "attention",
    emoji: "⚠️",
    color: "text-error",
  },
  {
    label: "Scheduled",
    key: "scheduled",
    filter: "scheduled",
    emoji: "📅",
    color: "text-warning",
  },
  {
    label: "Idle",
    key: "idle",
    filter: "idle",
    emoji: "💤",
    color: "text-foreground-subtle",
  },
];

export function StatsGrid({ summary, activeTab, onTabChange }: Props) {
  const isBillingEnabled = useIsBillingEnabled();
  const tiles = isBillingEnabled
    ? TILES
    : TILES.filter((tile) => tile.key !== "monthlySpend");

  return (
    <div
      className={cn(
        "grid grid-cols-1 gap-3 min-[450px]:grid-cols-2 sm:grid-cols-3",
        isBillingEnabled ? "lg:grid-cols-6" : "lg:grid-cols-5",
      )}
    >
      {tiles.map((tile) => {
        const rawValue = summary[tile.key];
        const value = tile.format ? tile.format(rawValue) : rawValue;
        const isActive = activeTab === tile.filter;
        // Without the scheduler a scheduled workflow can't be told from an
        // idle one, so neither count is shown rather than a wrong one.
        const isUnknown =
          summary.schedulesUnknown &&
          (tile.key === "scheduled" || tile.key === "idle");

        return (
          <button
            key={tile.label}
            type="button"
            aria-pressed={isActive}
            // A second click goes back to the overview, which has no tile of
            // its own when billing is off.
            onClick={() => onTabChange(isActive ? "all" : tile.filter)}
            title={
              isUnknown ? "Schedules are temporarily unavailable" : undefined
            }
            className={cn(
              "flex min-w-0 flex-col gap-1 rounded-medium border p-3 text-left shadow-md transition-all hover:shadow-lg",
              isActive
                ? "border-action-hover bg-surface-muted"
                : "border-separator bg-surface",
            )}
          >
            <div className="flex min-w-0 items-center gap-1.5">
              <Emoji text={tile.emoji} size={18} />
              <OverflowText value={tile.label} variant="body" tone="emphasis" />
            </div>
            <Text variant="h4">
              {isUnknown ? (
                <>
                  <span aria-hidden="true">—</span>
                  <span className="sr-only">Unknown</span>
                </>
              ) : (
                value
              )}
            </Text>
          </button>
        );
      })}
    </div>
  );
}
