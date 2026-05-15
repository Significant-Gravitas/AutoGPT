"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { formatCents } from "@/app/(platform)/copilot/components/usageHelpers";
import { DailySpendBars } from "./components/DailySpendBars";
import { SpendByAgentList } from "./components/SpendByAgentList";
import { TopRunsList } from "./components/TopRunsList";
import { useCostsBreakdown } from "./useCostsBreakdown";

interface Props {
  agents: LibraryAgent[];
}

export function CostsBreakdown({ agents }: Props) {
  const { summary, agentLookup, isLoading, isError, hasAnySpend } =
    useCostsBreakdown(agents);

  if (isError) return null;

  if (isLoading || !summary) {
    return (
      <section className="mt-6 flex flex-col gap-3">
        <Skeleton className="h-5 w-32" />
        <Skeleton className="h-16 w-full" />
        <Skeleton className="h-16 w-full" />
      </section>
    );
  }

  if (!hasAnySpend) {
    return (
      <section className="mt-6">
        <Text variant="body-medium" className="text-neutral-700">
          Cost breakdown
        </Text>
        <Text variant="body" className="mt-1 text-neutral-500">
          No spend this month yet.
        </Text>
      </section>
    );
  }

  const avgCostCents =
    summary.run_count > 0 ? summary.total_cents / summary.run_count : 0;

  return (
    <section className="mt-6 flex flex-col gap-4">
      <Text variant="body-medium" className="text-neutral-800">
        Cost breakdown
      </Text>

      <StatRow
        items={[
          {
            label: "Total this month",
            value: formatCents(summary.total_cents),
          },
          { label: "Runs", value: String(summary.run_count) },
          { label: "Avg / run", value: formatCents(Math.round(avgCostCents)) },
          {
            label: "Wasted on failed",
            value: formatCents(summary.failed_cost_cents),
            tone: summary.failed_cost_cents > 0 ? "warn" : undefined,
          },
        ]}
      />

      <TopRunsList runs={summary.top_runs} agentLookup={agentLookup} />
      <SpendByAgentList rollups={summary.by_agent} agentLookup={agentLookup} />
      <DailySpendBars daily={summary.daily} />
    </section>
  );
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
