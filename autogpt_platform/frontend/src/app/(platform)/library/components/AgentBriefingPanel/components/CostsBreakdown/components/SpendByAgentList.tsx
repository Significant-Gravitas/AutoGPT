"use client";

import type { UserAgentCostRollup } from "@/app/api/__generated__/models/userAgentCostRollup";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { formatCents } from "@/app/(platform)/copilot/components/usageHelpers";
import Link from "next/link";
import { useState } from "react";
import type { AgentLookupEntry } from "../helpers";

interface Props {
  rollups: UserAgentCostRollup[];
  agentLookup: Map<string, AgentLookupEntry>;
  totalCents: number;
}

const INITIAL_VISIBLE = 5;

export function SpendByAgentList({ rollups, agentLookup, totalCents }: Props) {
  const [showAll, setShowAll] = useState(false);

  const billable = rollups.filter((r) => r.cost_cents > 0);
  if (billable.length === 0) return null;

  const visible = showAll ? billable : billable.slice(0, INITIAL_VISIBLE);
  const hasMore = billable.length > INITIAL_VISIBLE;

  return (
    <section className="flex flex-col gap-2 lg:mt-[1rem]">
      <div className="flex items-baseline justify-between gap-2 lg:mb-[.5rem]">
        <Text variant="body-medium" className="text-neutral-800">
          Spend by agent
        </Text>
        <Text variant="small" className="text-neutral-400">
          % of monthly spend
        </Text>
      </div>
      <ul className="flex flex-col gap-4">
        {visible.map((rollup) => {
          const agent = agentLookup.get(rollup.graph_id);
          const label = agent?.name ?? `Agent ${rollup.graph_id.slice(0, 8)}`;
          const share = totalCents > 0 ? rollup.cost_cents / totalCents : 0;
          const sharePct = Math.round(share * 100);
          const widthPct = Math.max(2, sharePct);
          const shareLabel =
            share > 0 && sharePct === 0 ? "<1%" : `${sharePct}%`;
          const avgPerRun =
            rollup.run_count > 0 ? rollup.cost_cents / rollup.run_count : 0;
          const href = agent ? `/library/agents/${agent.libraryAgentId}` : null;

          const row = (
            <div className="flex flex-col gap-1">
              <div className="flex items-baseline justify-between gap-3">
                <Text variant="body" className="truncate text-neutral-800">
                  {label}
                </Text>
                <Text
                  variant="body-medium"
                  className="tabular-nums text-neutral-800"
                >
                  {formatCents(rollup.cost_cents)}
                  <Text
                    variant="small"
                    as="span"
                    className="ml-2 text-neutral-400"
                  >
                    {rollup.run_count} run{rollup.run_count === 1 ? "" : "s"}
                    {avgPerRun > 0 &&
                      ` · avg ${formatCents(Math.round(avgPerRun))}`}
                  </Text>
                </Text>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-100">
                  <div
                    className="h-full rounded-full bg-blue-500"
                    style={{ width: `${widthPct}%` }}
                    aria-hidden="true"
                  />
                </div>
                <Text
                  variant="small"
                  className="w-9 shrink-0 text-right tabular-nums text-neutral-400"
                >
                  {shareLabel}
                </Text>
              </div>
            </div>
          );

          return (
            <li key={rollup.graph_id}>
              {href ? (
                <Link
                  href={href}
                  className="-m-1 block rounded-small p-1 hover:bg-zinc-50"
                >
                  {row}
                </Link>
              ) : (
                row
              )}
            </li>
          );
        })}
      </ul>
      {hasMore && (
        <div className="flex justify-center">
          <Button
            variant="secondary"
            size="small"
            onClick={() => setShowAll(!showAll)}
          >
            {showAll ? "Collapse" : `Show all (${billable.length})`}
          </Button>
        </div>
      )}
    </section>
  );
}
