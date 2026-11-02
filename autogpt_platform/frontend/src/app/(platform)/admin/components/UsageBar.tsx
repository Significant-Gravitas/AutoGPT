"use client";

import type { CSSProperties } from "react";
import { formatMicrodollarsAsUsd } from "@/app/(platform)/copilot/components/usageHelpers";

export function UsageBar({ used, limit }: { used: number; limit: number }) {
  if (limit === 0) {
    return <span className="text-sm text-foreground-tertiary">Unlimited</span>;
  }
  const pct = Math.min(Math.max(0, (used / limit) * 100), 100);
  const color =
    // oxlint-disable-next-line design-system/no-raw-colors -- yellow has no role; warning amber would change the hue (deferred)
    pct >= 90 ? "bg-error" : pct >= 70 ? "bg-yellow-500" : "bg-success";

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span>{formatMicrodollarsAsUsd(used)} spent</span>
        <span>{formatMicrodollarsAsUsd(limit)} limit</span>
      </div>
      <div className="h-2 w-full rounded-full bg-surface-active">
        <div
          className={`h-2 w-[var(--usage-width)] rounded-full ${color}`}
          style={{ "--usage-width": `${pct}%` } as CSSProperties}
        />
      </div>
      <div className="text-right text-xs text-foreground-tertiary">
        {pct.toFixed(1)}% used
      </div>
    </div>
  );
}
