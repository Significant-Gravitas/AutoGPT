"use client";

import type { UserTopRun } from "@/app/api/__generated__/models/userTopRun";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { formatCents } from "@/app/(platform)/copilot/components/usageHelpers";
import Link from "next/link";
import { useState } from "react";
import { formatRelativeDate, type AgentLookupEntry } from "../helpers";

interface Props {
  runs: UserTopRun[];
  agentLookup: Map<string, AgentLookupEntry>;
}

const INITIAL_VISIBLE = 5;

export function TopRunsList({ runs, agentLookup }: Props) {
  const [showAll, setShowAll] = useState(false);
  if (runs.length === 0) return null;

  const visible = showAll ? runs : runs.slice(0, INITIAL_VISIBLE);
  const hasMore = runs.length > INITIAL_VISIBLE;

  return (
    <section className="flex flex-col gap-2 lg:mt-[1rem]">
      <Text variant="body-medium" className="text-neutral-800 lg:mb-[.5rem]">
        Most expensive tasks
      </Text>
      <ul className="flex flex-col divide-y divide-zinc-100 rounded-medium border border-zinc-100 bg-white">
        {visible.map((run) => {
          const agent = agentLookup.get(run.graph_id);
          const label = agent?.name ?? `Agent ${run.graph_id.slice(0, 8)}`;
          const href = agent
            ? `/library/agents/${agent.libraryAgentId}?activeItem=${run.execution_id}`
            : null;

          const row = (
            <div className="flex items-center justify-between gap-3 px-3 py-2">
              <div className="min-w-0 flex-1">
                <Text variant="body" className="truncate text-neutral-800">
                  {label}
                </Text>
                <Text variant="small" className="text-neutral-400">
                  {formatRelativeDate(run.started_at)}
                  {run.node_error_count > 0
                    ? ` · ${run.node_error_count} error${run.node_error_count === 1 ? "" : "s"}`
                    : ""}
                </Text>
              </div>
              <Text
                variant="body-medium"
                className="tabular-nums text-neutral-800"
              >
                {formatCents(run.cost_cents)}
              </Text>
            </div>
          );

          return (
            <li key={run.execution_id}>
              {href ? (
                <Link href={href} className="block hover:bg-zinc-50">
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
            {showAll ? "Collapse" : `Show all (${runs.length})`}
          </Button>
        </div>
      )}
    </section>
  );
}
