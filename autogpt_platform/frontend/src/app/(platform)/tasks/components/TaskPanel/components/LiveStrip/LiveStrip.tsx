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
      className="flex flex-col gap-1.5 rounded-card bg-surface px-3.5 py-3 shadow-card"
    >
      <div className="flex items-center">
        <StepRow
          status="running"
          label={live.current_step ?? "Getting started"}
          trailing={<LiveTimer since={live.started_at} />}
        />
      </div>
      <div className="pl-6">
        <Text variant="small" tone="quiet" tabular>
          {formatCost(metrics.cost_usd)} · {formatTokens(metrics.tokens)} ·{" "}
          {metrics.tool_calls} tool call{metrics.tool_calls === 1 ? "" : "s"}
        </Text>
      </div>
    </section>
  );
}
