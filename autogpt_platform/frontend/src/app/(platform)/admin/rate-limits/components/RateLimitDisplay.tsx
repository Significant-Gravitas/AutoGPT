"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import type { UserRateLimitResponse } from "@/app/api/__generated__/models/userRateLimitResponse";

function formatTokens(tokens: number): string {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
  if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(0)}K`;
  return tokens.toString();
}

function UsageBar({ used, limit }: { used: number; limit: number }) {
  if (limit === 0) {
    return <span className="text-sm text-gray-500">Unlimited</span>;
  }
  const pct = Math.min((used / limit) * 100, 100);
  const color =
    pct >= 90 ? "bg-red-500" : pct >= 70 ? "bg-yellow-500" : "bg-green-500";

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span>{formatTokens(used)} used</span>
        <span>{limit === 0 ? "Unlimited" : formatTokens(limit)} limit</span>
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

interface Props {
  data: UserRateLimitResponse;
  onReset: () => Promise<void>;
}

export function RateLimitDisplay({ data, onReset }: Props) {
  const [isResetting, setIsResetting] = useState(false);

  async function handleReset() {
    setIsResetting(true);
    try {
      await onReset();
    } finally {
      setIsResetting(false);
    }
  }

  return (
    <div className="rounded-md border bg-white p-6">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold">
          Rate Limits for {data.user_id}
        </h2>
        <Button
          variant="outline"
          onClick={handleReset}
          disabled={
            isResetting ||
            (data.daily_tokens_used === 0 && data.weekly_tokens_used === 0)
          }
        >
          {isResetting ? "Resetting..." : "Reset Usage to Zero"}
        </Button>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700">Daily Usage</h3>
          <UsageBar
            used={data.daily_tokens_used}
            limit={data.daily_token_limit}
          />
        </div>
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700">Weekly Usage</h3>
          <UsageBar
            used={data.weekly_tokens_used}
            limit={data.weekly_token_limit}
          />
        </div>
      </div>
    </div>
  );
}
