"use client";

import { formatMicrodollarsAsUsd } from "@/app/(platform)/copilot/components/usageHelpers";

export function UsageBar({ used, limit }: { used: number; limit: number }) {
  if (limit === 0) {
    return <span className="text-sm text-gray-500">Unlimited</span>;
  }
  const pct = Math.min(Math.max(0, (used / limit) * 100), 100);
  const color =
    pct >= 90 ? "bg-red-500" : pct >= 70 ? "bg-yellow-500" : "bg-green-500";

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span>{formatMicrodollarsAsUsd(used)} spent</span>
        <span>{formatMicrodollarsAsUsd(limit)} limit</span>
      </div>
      <div className="h-2 w-full rounded-full bg-gray-200">
        <div
          className={`h-2 rounded-full ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="text-right text-xs text-gray-500">
        {pct.toFixed(1)}% used
      </div>
    </div>
  );
}
