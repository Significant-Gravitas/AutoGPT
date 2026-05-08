"use client";

import { useState } from "react";
import { useToast } from "@/components/molecules/Toast/use-toast";
import type { SetUserTierRequest } from "@/app/api/__generated__/models/setUserTierRequest";
import type { UserRateLimitResponse } from "@/app/api/__generated__/models/userRateLimitResponse";
import {
  getV2GetUserRateLimit,
  getV2SearchUsersByNameOrEmail,
  postV2ResetUserRateLimitUsage,
  postV2SetUserRateLimitTier,
} from "@/app/api/__generated__/endpoints/admin/admin";

export interface UserOption {
  user_id: string;
  user_email: string;
}

function looksLikeEmail(input: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(input);
}

function looksLikeUuid(input: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
    input,
  );
}

export function useRateLimitManager() {
  const { toast } = useToast();
  const [isSearching, setIsSearching] = useState(false);
  const [isLoadingRateLimit, setIsLoadingRateLimit] = useState(false);
  const [searchResults, setSearchResults] = useState<UserOption[]>([]);
  const [selectedUser, setSelectedUser] = useState<UserOption | null>(null);
  const [rateLimitData, setRateLimitData] =
    useState<UserRateLimitResponse | null>(null);

  async function handleDirectLookup(trimmed: string) {
    setIsSearching(true);
    setSearchResults([]);
    setSelectedUser(null);
    setRateLimitData(null);

    try {
      const params = looksLikeEmail(trimmed)
        ? { email: trimmed }
        : { user_id: trimmed };
      const response = await getV2GetUserRateLimit(params);
      if (response.status !== 200) {
        throw new Error("Failed to fetch rate limit");
      }
      setRateLimitData(response.data);
      setSelectedUser({
        user_id: response.data.user_id,
        user_email: response.data.user_email ?? response.data.user_id,
      });
    } catch (error) {
      console.error("Error fetching rate limit:", error);
      const hint = looksLikeEmail(trimmed)
        ? "No user found with that email address."
        : "Check the user ID and try again.";
      toast({
        title: "Error",
        description: `Failed to fetch rate limits. ${hint}`,
        variant: "destructive",
      });
      setRateLimitData(null);
    } finally {
      setIsSearching(false);
    }
  }

  async function handleFuzzySearch(trimmed: string) {
    setIsSearching(true);
    setSearchResults([]);
    setSelectedUser(null);
    setRateLimitData(null);

    try {
      const response = await getV2SearchUsersByNameOrEmail({
        query: trimmed,
        limit: 20,
      });
      if (response.status !== 200) {
        throw new Error("Failed to search users");
      }

      const users = (response.data ?? []).map((u) => ({
        user_id: u.user_id,
        user_email: u.user_email ?? u.user_id,
      }));
      if (users.length === 0) {
        toast({ title: "No results", description: "No users found." });
      }
      setSearchResults(users);
    } catch (error) {
      console.error("Error searching users:", error);
      toast({
        title: "Error",
        description: "Failed to search users.",
        variant: "destructive",
      });
    } finally {
      setIsSearching(false);
    }
  }

  async function handleSearch(query: string) {
    const trimmed = query.trim();
    if (!trimmed) return;

    // Direct lookup when the input is a full email or UUID.
    // This avoids the spending-history indirection and works even for
    // users who have no credit transaction history.
    if (looksLikeEmail(trimmed) || looksLikeUuid(trimmed)) {
      await handleDirectLookup(trimmed);
    } else {
      await handleFuzzySearch(trimmed);
    }
  }

  async function fetchRateLimit(userId: string) {
    setIsLoadingRateLimit(true);
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
      setIsLoadingRateLimit(false);
    }
  }

  async function handleSelectUser(user: UserOption) {
    setSelectedUser(user);
    setRateLimitData(null);
    await fetchRateLimit(user.user_id);
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

  async function handleTierChange(newTier: string) {
    if (!rateLimitData) return;

    const response = await postV2SetUserRateLimitTier({
      user_id: rateLimitData.user_id,
      tier: newTier as SetUserTierRequest["tier"],
    });

    if (response.status !== 200) {
      throw new Error("Failed to update tier");
    }

    // Re-fetch rate limit data to reflect new tier-adjusted limits.
    try {
      const refreshResponse = await getV2GetUserRateLimit({
        user_id: rateLimitData.user_id,
      });
      if (refreshResponse.status === 200) {
        setRateLimitData(refreshResponse.data);
      }
    } catch {
      // Tier was changed server-side; UI will be stale but not incorrect.
      // The caller's success toast is still valid — the tier change worked.
    }
  }

  return {
    isSearching,
    isLoadingRateLimit,
    searchResults,
    selectedUser,
    rateLimitData,
    handleSearch,
    handleSelectUser,
    handleReset,
    handleTierChange,
  };
}
