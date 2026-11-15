"use client";

import type { CSSProperties } from "react";
import { formatMicrodollarsAsUsd } from "@/app/(platform)/copilot/components/usageHelpers";

export function UsageBar({ used, limit }: { used: number; limit: number }) {
  // The backend encodes both "no cap" and "no spend allowed" as non-dollar
  // values: a negative limit disables the window's cap (the self-hosted
  // default) and 0 blocks all spend (see ChatConfig.daily_cost_limit_microdollars
  // and weekly_cost_limit_microdollars). Neither one is a cap to divide by, so
  // show the spend next to what the limit actually means.
  if (limit <= 0) {
    return (
      <div className="flex justify-between text-body">
        <span>{formatMicrodollarsAsUsd(used)} spent</span>
        <span className="text-foreground-tertiary">
          {limit < 0 ? "Unlimited" : "No allowance"}
        </span>
      </div>
    );
  }
  const pct = Math.min(Math.max(0, (used / limit) * 100), 100);
  const color = pct >= 90 ? "bg-error" : pct >= 70 ? "bg-yellow" : "bg-success";

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-body">
        <span>{formatMicrodollarsAsUsd(used)} spent</span>
        <span>{formatMicrodollarsAsUsd(limit)} limit</span>
      </div>
      <div className="h-2 w-full rounded-full bg-surface-active">
        <div
          className={`h-2 w-[var(--usage-width)] rounded-full ${color}`}
          style={{ "--usage-width": `${pct}%` } as CSSProperties}
        />
      </div>
      <div className="text-right text-caption text-foreground-tertiary">
        {pct.toFixed(1)}% used
      </div>
    </div>
  );
}
