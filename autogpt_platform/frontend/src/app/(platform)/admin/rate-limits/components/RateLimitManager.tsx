"use client";

import { useState } from "react";
import { Button } from "@/components/__legacy__/ui/button";
import { Input } from "@/components/__legacy__/ui/input";
import { Label } from "@/components/__legacy__/ui/label";
import { Search } from "lucide-react";
import { useToast } from "@/components/molecules/Toast/use-toast";
import type { UserRateLimitResponse } from "@/lib/autogpt-server-api/types";
import { getUserRateLimit, resetUserRateLimit } from "../actions";
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
      const data = await getUserRateLimit(trimmed);
      setRateLimitData(data);
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

  async function handleReset() {
    if (!rateLimitData) return;

    try {
      const data = await resetUserRateLimit(rateLimitData.user_id);
      setRateLimitData(data);
      toast({
        title: "Success",
        description: "User rate limit usage reset to zero.",
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
      <div className="rounded-md border bg-white p-6">
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
            {isLoading ? "Loading..." : <Search className="h-4 w-4" />}
          </Button>
        </div>
      </div>

      {rateLimitData && (
        <RateLimitDisplay data={rateLimitData} onReset={handleReset} />
      )}
    </div>
  );
}
