"use client";

import { useState, useEffect } from "react";
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
import { RateLimitDisplay } from "../../rate-limits/components/RateLimitDisplay";

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

  useEffect(() => {
    if (!open) {
      setRateLimitData(null);
      return;
    }

    async function fetchRateLimit() {
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
    }

    fetchRateLimit();
  }, [open, userId, toast]);

  async function handleReset(resetWeekly: boolean) {
    if (!rateLimitData) return;

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
    }
  }

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
              CoPilot rate limits for {userEmail || userId}
            </DialogDescription>
          </DialogHeader>

          {isLoading && (
            <div className="py-8 text-center text-gray-500">
              Loading rate limits...
            </div>
          )}

          {!isLoading && rateLimitData && (
            <RateLimitDisplay
              data={rateLimitData}
              onReset={handleReset}
              className="space-y-4"
            />
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
