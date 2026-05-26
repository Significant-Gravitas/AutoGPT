"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { formatCents } from "@/app/(platform)/copilot/components/usageHelpers";
import { CaretDownIcon } from "@phosphor-icons/react";
import { useState } from "react";
import { SpendByAgentList } from "./components/SpendByAgentList";
import { TopRunsList } from "./components/TopRunsList";
import { useCostsBreakdown } from "./useCostsBreakdown";

interface Props {
  agents: LibraryAgent[];
}

export function CostsBreakdown({ agents }: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  const { summary, agentLookup, isLoading, isError, hasAnySpend } =
    useCostsBreakdown(agents, { enabled: isExpanded });

  return (
    <section className="mt-6 flex flex-col gap-4">
      <Button
        variant="ghost"
        size="small"
        onClick={() => setIsExpanded((prev) => !prev)}
        aria-expanded={isExpanded}
        className="w-fit gap-1 px-0 text-neutral-800 hover:bg-transparent"
      >
        {isExpanded ? "Hide costs breakdown" : "See costs breakdown"}
        <CaretDownIcon
          size={14}
          className={`transition-transform ${isExpanded ? "rotate-180" : ""}`}
        />
      </Button>

      {isExpanded && (
        <ExpandedBody
          summary={summary}
          agentLookup={agentLookup}
          isLoading={isLoading}
          isError={isError}
          hasAnySpend={hasAnySpend}
        />
      )}
    </section>
  );
}

interface ExpandedBodyProps {
  summary: ReturnType<typeof useCostsBreakdown>["summary"];
  agentLookup: ReturnType<typeof useCostsBreakdown>["agentLookup"];
  isLoading: boolean;
  isError: boolean;
  hasAnySpend: boolean;
}

function ExpandedBody({
  summary,
  agentLookup,
  isLoading,
  isError,
  hasAnySpend,
}: ExpandedBodyProps) {
  if (isError) {
    return (
      <Text variant="body" className="text-neutral-500">
        Couldn&apos;t load cost breakdown.
      </Text>
    );
  }

  if (isLoading || !summary) {
    return (
      <div className="flex flex-col gap-3">
        <Skeleton className="h-16 w-full" />
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <Skeleton className="h-48 w-full" />
          <Skeleton className="h-48 w-full" />
        </div>
      </div>
    );
  }

  if (!hasAnySpend) {
    return (
      <Text variant="body" className="text-neutral-500">
        No spend this month yet.
      </Text>
    );
  }

  // Divide by billable_run_count (runs with cost>0), not run_count, so the
  // average isn't deflated by zero-cost runs that contribute to the count
  // but nothing to the total.
  const avgCostCents =
    summary.billable_run_count > 0
      ? summary.total_cents / summary.billable_run_count
      : 0;

  return (
    <>
      <Text variant="small" className="text-neutral-500">
        Calendar month so far · {formatMonthRangeLabel()}
      </Text>
      <StatRow
        items={[
          {
            label: "Total this month",
            value: formatCents(summary.total_cents),
          },
          { label: "Tasks", value: String(summary.run_count) },
          {
            label: "Avg / task",
            value: formatCents(Math.round(avgCostCents)),
          },
          {
            label: "Wasted on failed",
            value: formatCents(summary.failed_cost_cents),
            tone: summary.failed_cost_cents > 0 ? "warn" : undefined,
          },
        ]}
      />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <TopRunsList runs={summary.top_runs} agentLookup={agentLookup} />
        <SpendByAgentList
          rollups={summary.by_agent}
          agentLookup={agentLookup}
          totalCents={summary.total_cents}
        />
      </div>
    </>
  );
}

function formatMonthRangeLabel(): string {
  const now = new Date();
  const monthStart = new Date(
    Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1),
  );
  const fmt = (d: Date) =>
    d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      timeZone: "UTC",
    });
  return `${fmt(monthStart)} – today (UTC)`;
}

interface StatItem {
  label: string;
  value: string;
  tone?: "warn";
}

function StatRow({ items }: { items: StatItem[] }) {
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {items.map((item) => (
        <div
          key={item.label}
          className="flex flex-col gap-0.5 rounded-medium border border-zinc-100 bg-white p-3"
        >
          <Text variant="small" className="text-neutral-500">
            {item.label}
          </Text>
          <Text
            variant="body-medium"
            className={
              item.tone === "warn"
                ? "tabular-nums text-orange-600"
                : "tabular-nums text-neutral-800"
            }
          >
            {item.value}
          </Text>
        </div>
      ))}
    </div>
  );
}
