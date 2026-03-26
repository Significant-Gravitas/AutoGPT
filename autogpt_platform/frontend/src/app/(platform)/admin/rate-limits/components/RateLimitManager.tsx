"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/__legacy__/ui/input";
import { Label } from "@/components/__legacy__/ui/label";
import { MagnifyingGlass } from "@phosphor-icons/react";
import { useToast } from "@/components/molecules/Toast/use-toast";
import type { UserRateLimitResponse } from "@/app/api/__generated__/models/userRateLimitResponse";
import {
  getV2GetUserRateLimit,
  postV2ResetUserRateLimitUsage,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { RateLimitDisplay } from "./RateLimitDisplay";

export function RateLimitManager() {
  const { toast } = useToast();
  const [userIdInput, setUserIdInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [rateLimitData, setRateLimitData] =
    useState<UserRateLimitResponse | null>(null);

  async function handleLookup() {
    const trimmed = userIdInput.trim();
    if (!trimmed) return;

    setIsLoading(true);
    try {
      const response = await getV2GetUserRateLimit({ user_id: trimmed });
      if (response.status !== 200) {
        throw new Error("Failed to fetch rate limit");
      }
      setRateLimitData(response.data);
    } catch (error) {
      console.error("Error fetching rate limit:", error);
      toast({
        title: "Error",
        description: "Failed to fetch user rate limit. Check the user ID.",
        variant: "destructive",
      });
      setRateLimitData(null);
    } finally {
      setIsLoading(false);
    }
  }

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
    <div className="space-y-6">
      <div className="rounded-md border bg-white p-6 dark:bg-gray-900">
        <Label htmlFor="userId" className="mb-2 block text-sm font-medium">
          User ID
        </Label>
        <div className="flex items-center gap-2">
          <Input
            id="userId"
            placeholder="Enter user ID to look up rate limits..."
            value={userIdInput}
            onChange={(e) => setUserIdInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleLookup()}
          />
          <Button
            variant="outline"
            onClick={handleLookup}
            disabled={isLoading || !userIdInput.trim()}
          >
            {isLoading ? "Loading..." : <MagnifyingGlass size={16} />}
          </Button>
        </div>
      </div>

      {rateLimitData && (
        <RateLimitDisplay data={rateLimitData} onReset={handleReset} />
      )}
    </div>
  );
}
