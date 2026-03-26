"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/atoms/Button/Button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/__legacy__/ui/dialog";
import { useToast } from "@/components/molecules/Toast/use-toast";
import type { UserRateLimitResponse } from "@/app/api/__generated__/models/userRateLimitResponse";
import {
  getV2GetUserRateLimit,
  postV2ResetUserRateLimitUsage,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { Gauge } from "@phosphor-icons/react";

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
        <span>{formatTokens(limit)} limit</span>
      </div>
      <div className="h-2 w-full rounded-full bg-gray-200 dark:bg-gray-700">
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

export function RateLimitModal({
  userId,
  userEmail,
}: {
  userId: string;
  userEmail: string;
}) {
  const { toast } = useToast();
  const [open, setOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [rateLimitData, setRateLimitData] =
    useState<UserRateLimitResponse | null>(null);
  const [resetWeekly, setResetWeekly] = useState(false);
  const [isResetting, setIsResetting] = useState(false);

  const fetchRateLimit = useCallback(async () => {
    setIsLoading(true);
    try {
      const response = await getV2GetUserRateLimit({ user_id: userId });
      if (response.status !== 200) {
        throw new Error("Failed to fetch rate limit");
      }
      setRateLimitData(response.data);
    } catch (error) {
      console.error("Error fetching rate limit:", error);
      toast({
        title: "Error",
        description: "Failed to fetch user rate limit.",
        variant: "destructive",
      });
      setRateLimitData(null);
    } finally {
      setIsLoading(false);
    }
  }, [userId, toast]);

  useEffect(() => {
    if (open) {
      fetchRateLimit();
    } else {
      setRateLimitData(null);
      setResetWeekly(false);
    }
  }, [open, fetchRateLimit]);

  async function handleReset() {
    if (!rateLimitData) return;

    const msg = resetWeekly
      ? "Reset both daily and weekly usage counters to zero?"
      : "Reset daily usage counter to zero?";
    if (!window.confirm(msg)) return;

    setIsResetting(true);
    try {
      const response = await postV2ResetUserRateLimitUsage({
        user_id: rateLimitData.user_id,
        reset_weekly: resetWeekly,
      });
      if (response.status !== 200) {
        throw new Error("Failed to reset usage");
      }
      setRateLimitData(response.data);
      toast({
        title: "Success",
        description: resetWeekly
          ? "Daily and weekly usage reset to zero."
          : "Daily usage reset to zero.",
      });
    } catch (error) {
      console.error("Error resetting rate limit:", error);
      toast({
        title: "Error",
        description: "Failed to reset rate limit usage.",
        variant: "destructive",
      });
    } finally {
      setIsResetting(false);
    }
  }

  const nothingToReset = rateLimitData
    ? resetWeekly
      ? rateLimitData.daily_tokens_used === 0 &&
        rateLimitData.weekly_tokens_used === 0
      : rateLimitData.daily_tokens_used === 0
    : true;

  return (
    <>
      <Button
        size="small"
        variant="outline"
        onClick={(e) => {
          e.stopPropagation();
          setOpen(true);
        }}
      >
        <Gauge size={16} className="mr-1" />
        Rate Limits
      </Button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Rate Limits</DialogTitle>
            <DialogDescription>
              CoPilot rate limits for {userEmail}
            </DialogDescription>
          </DialogHeader>

          {isLoading && (
            <div className="py-8 text-center text-gray-500">
              Loading rate limits...
            </div>
          )}

          {!isLoading && rateLimitData && (
            <div className="space-y-6">
              <div className="grid grid-cols-2 gap-6">
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Daily Usage
                  </h3>
                  <UsageBar
                    used={rateLimitData.daily_tokens_used}
                    limit={rateLimitData.daily_token_limit}
                  />
                </div>
                <div className="space-y-2">
                  <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Weekly Usage
                  </h3>
                  <UsageBar
                    used={rateLimitData.weekly_tokens_used}
                    limit={rateLimitData.weekly_token_limit}
                  />
                </div>
              </div>

              <div className="flex items-center gap-3 border-t pt-4">
                <select
                  value={resetWeekly ? "both" : "daily"}
                  onChange={(e) => setResetWeekly(e.target.value === "both")}
                  className="rounded-md border bg-white px-3 py-1.5 text-sm dark:bg-gray-800 dark:text-gray-200"
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
          )}

          {!isLoading && !rateLimitData && (
            <div className="py-8 text-center text-gray-500">
              No rate limit data available for this user.
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
