"use client";

import { Loading03Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { formatCost, formatTokens } from "../../../../data/helpers";
import type { TaskLiveStrip, TaskMetrics } from "../../../../data/types";
import { formatElapsed } from "../../helpers";
import { useLiveStrip } from "./useLiveStrip";

interface Props {
  live: TaskLiveStrip;
  metrics: TaskMetrics;
}

export function LiveStrip({ live, metrics }: Props) {
  const { now } = useLiveStrip();

  return (
    <section
      aria-label="Live progress"
      className="flex flex-col gap-2 rounded-xl border border-border-base bg-info-surface p-4"
    >
      <div className="flex items-center gap-2">
        <Icon icon={Loading03Icon} tone="info" spin aria-hidden />
        <Text variant="small-semibold" tone="info-strong">
          Working
        </Text>
        <Text variant="small" tone="muted" tabular className="ml-auto">
          {formatElapsed(live.started_at, now)}
        </Text>
      </div>
      <Text variant="body-tight" tone="primary">
        {live.current_step ?? "Getting started"}
      </Text>
      <Text variant="small" tone="muted" tabular>
        {formatCost(metrics.cost_usd)} · {formatTokens(metrics.tokens)} ·{" "}
        {metrics.tool_calls} tool call{metrics.tool_calls === 1 ? "" : "s"}
      </Text>
    </section>
  );
}
