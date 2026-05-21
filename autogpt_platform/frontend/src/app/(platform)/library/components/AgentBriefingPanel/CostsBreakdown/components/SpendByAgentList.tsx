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
}

const INITIAL_VISIBLE = 5;

export function SpendByAgentList({ rollups, agentLookup }: Props) {
  const [showAll, setShowAll] = useState(false);
  if (rollups.length === 0) return null;

  const maxCost = Math.max(...rollups.map((r) => r.cost_cents), 1);
  const visible = showAll ? rollups : rollups.slice(0, INITIAL_VISIBLE);
  const hasMore = rollups.length > INITIAL_VISIBLE;

  return (
    <section className="flex flex-col gap-2 lg:mt-[1rem]">
      <Text variant="body-medium" className="text-neutral-800 lg:mb-[.5rem]">
        Spend by agent
      </Text>
      <ul className="flex flex-col gap-4">
        {visible.map((rollup) => {
          const agent = agentLookup.get(rollup.graph_id);
          const label = agent?.name ?? `Agent ${rollup.graph_id.slice(0, 8)}`;
          const widthPct = Math.max(
            2,
            Math.round((rollup.cost_cents / maxCost) * 100),
          );
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
                  </Text>
                </Text>
              </div>
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-100">
                <div
                  className="h-full rounded-full bg-blue-500"
                  style={{ width: `${widthPct}%` }}
                  aria-hidden="true"
                />
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
            {showAll ? "Collapse" : `Show all (${rollups.length})`}
          </Button>
        </div>
      )}
    </section>
  );
}
