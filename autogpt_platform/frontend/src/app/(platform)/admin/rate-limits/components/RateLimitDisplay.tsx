"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import type { UserRateLimitResponse } from "@/app/api/__generated__/models/userRateLimitResponse";
import { UsageBar } from "../../components/UsageBar";

interface Props {
  data: UserRateLimitResponse;
  onReset: (resetWeekly: boolean) => Promise<void>;
  /** Override the outer container classes (default: bordered card). */
  className?: string;
}

export function RateLimitDisplay({ data, onReset, className }: Props) {
  const [isResetting, setIsResetting] = useState(false);
  const [resetWeekly, setResetWeekly] = useState(false);

  async function handleReset() {
    const msg = resetWeekly
      ? "Reset both daily and weekly usage counters to zero?"
      : "Reset daily usage counter to zero?";
    if (!window.confirm(msg)) return;

    setIsResetting(true);
    try {
      await onReset(resetWeekly);
    } finally {
      setIsResetting(false);
    }
  }

  const nothingToReset = resetWeekly
    ? data.daily_tokens_used === 0 && data.weekly_tokens_used === 0
    : data.daily_tokens_used === 0;

  return (
    <div className={className ?? "rounded-md border bg-white p-6"}>
      <h2 className="mb-1 text-lg font-semibold">
        Rate Limits for {data.user_email ?? data.user_id}
      </h2>
      {data.user_email && (
        <p className="mb-4 text-xs text-gray-500">User ID: {data.user_id}</p>
      )}
      {!data.user_email && <div className="mb-4" />}

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

      <div className="mt-6 flex items-center gap-3 border-t pt-4">
        <select
          aria-label="Reset scope"
          value={resetWeekly ? "both" : "daily"}
          onChange={(e) => setResetWeekly(e.target.value === "both")}
          className="rounded-md border bg-white px-3 py-1.5 text-sm"
          disabled={isResetting}
        >
          <option value="daily">Reset daily only</option>
          <option value="both">Reset daily + weekly</option>
        </select>
        <Button
          variant="outline"
          onClick={handleReset}
          disabled={isResetting || nothingToReset}
        >
          {isResetting ? "Resetting..." : "Reset Usage"}
        </Button>
      </div>
    </div>
  );
}
