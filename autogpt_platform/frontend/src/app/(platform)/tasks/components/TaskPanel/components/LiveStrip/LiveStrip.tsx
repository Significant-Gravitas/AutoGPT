"use client";

import { Text } from "@/components/atoms/Text/Text";
import { formatCost, formatTokens } from "../../../../data/helpers";
import type { TaskLiveStrip, TaskMetrics } from "../../../../data/types";
import { LiveTimer } from "../LiveTimer/LiveTimer";
import { StepRow } from "../StepRow/StepRow";

interface Props {
  live: TaskLiveStrip;
  metrics: TaskMetrics;
}

export function LiveStrip({ live, metrics }: Props) {
  return (
    <section
      aria-label="Live progress"
      className="flex flex-col gap-2 rounded-xl border border-border-card bg-surface px-3 py-2.5 shadow-sm"
    >
      <div className="flex items-center">
        <StepRow
          status="running"
          label={live.current_step ?? "Getting started"}
          trailing={<LiveTimer since={live.started_at} />}
        />
      </div>
      <Text variant="small" tone="quiet" tabular className="pl-6">
        {formatCost(metrics.cost_usd)} · {formatTokens(metrics.tokens)} ·{" "}
        {metrics.tool_calls} tool call{metrics.tool_calls === 1 ? "" : "s"}
      </Text>
    </section>
  );
}
