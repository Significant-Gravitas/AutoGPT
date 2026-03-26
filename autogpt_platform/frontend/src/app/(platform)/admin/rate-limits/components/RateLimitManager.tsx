"use client";

import { useState } from "react";
import { Label } from "@/components/__legacy__/ui/label";
import { useToast } from "@/components/molecules/Toast/use-toast";
import type { UserRateLimitResponse } from "@/app/api/__generated__/models/userRateLimitResponse";
import type { UserTransaction } from "@/app/api/__generated__/models/userTransaction";
import {
  getV2GetUserRateLimit,
  getV2GetAllUsersHistory,
  postV2ResetUserRateLimitUsage,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { AdminUserSearch } from "../../components/AdminUserSearch";
import { RateLimitDisplay } from "./RateLimitDisplay";

interface UserOption {
  user_id: string;
  user_email: string;
}

/**
 * Returns true when the input looks like a complete email address.
 * Used to decide whether to call the direct email lookup endpoint
 * vs. the broader user-history search.
 */
function looksLikeEmail(input: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(input);
}

/**
 * Returns true when the input looks like a UUID (user ID).
 */
function looksLikeUuid(input: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
    input,
  );
}

export function RateLimitManager() {
  const { toast } = useToast();
  const [isSearching, setIsSearching] = useState(false);
  const [isLoadingRateLimit, setIsLoadingRateLimit] = useState(false);
  const [searchResults, setSearchResults] = useState<UserOption[]>([]);
  const [selectedUser, setSelectedUser] = useState<UserOption | null>(null);
  const [rateLimitData, setRateLimitData] =
    useState<UserRateLimitResponse | null>(null);

  /** Direct lookup by email or user ID via the rate-limit endpoint. */
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

  /** Fuzzy name/email search via the spending-history endpoint. */
  async function handleFuzzySearch(trimmed: string) {
    setIsSearching(true);
    setSearchResults([]);
    setSelectedUser(null);
    setRateLimitData(null);

    try {
      const response = await getV2GetAllUsersHistory({
        search: trimmed,
        page: 1,
        page_size: 50,
      });
      if (response.status !== 200) {
        throw new Error("Failed to search users");
      }

      // Deduplicate by user_id to get unique users
      const seen = new Set<string>();
      const users: UserOption[] = [];
      for (const tx of response.data.history as UserTransaction[]) {
        if (!seen.has(tx.user_id)) {
          seen.add(tx.user_id);
          users.push({
            user_id: tx.user_id,
            user_email: String(tx.user_email ?? tx.user_id),
          });
        }
      }

      if (users.length === 0) {
        toast({
          title: "No results",
          description: "No users found matching your search.",
        });
      } else if (users.length === 1) {
        // Auto-select if only one match
        setSelectedUser(users[0]);
        setSearchResults(users);
        await fetchRateLimit(users[0].user_id);
        return;
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

  return (
    <div className="space-y-6">
      <div className="rounded-md border bg-white p-6 dark:bg-gray-900">
        <Label className="mb-2 block text-sm font-medium">Search User</Label>
        <AdminUserSearch
          onSearch={handleSearch}
          placeholder="Search by name, email, or user ID..."
          isLoading={isSearching}
        />
        <p className="mt-1.5 text-xs text-gray-500">
          Exact email or user ID does a direct lookup. Partial text searches
          user history.
        </p>
      </div>

      {/* User selection list when multiple results are found */}
      {searchResults.length > 1 && !selectedUser && (
        <div className="rounded-md border bg-white p-4 dark:bg-gray-900">
          <h3 className="mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">
            Select a user ({searchResults.length} results)
          </h3>
          <ul className="divide-y">
            {searchResults.map((user) => (
              <li key={user.user_id}>
                <button
                  className="w-full px-2 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-800"
                  onClick={() => handleSelectUser(user)}
                >
                  <span className="font-medium">{user.user_email}</span>
                  <span className="ml-2 text-xs text-gray-500">
                    {user.user_id}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Show selected user */}
      {selectedUser && searchResults.length > 1 && (
        <div className="rounded-md border border-blue-200 bg-blue-50 px-4 py-2 text-sm dark:border-blue-800 dark:bg-blue-950">
          Selected:{" "}
          <span className="font-medium">{selectedUser.user_email}</span>
          <span className="ml-2 text-xs text-gray-500">
            {selectedUser.user_id}
          </span>
        </div>
      )}

      {isLoadingRateLimit && (
        <div className="py-4 text-center text-sm text-gray-500">
          Loading rate limits...
        </div>
      )}

      {rateLimitData && (
        <RateLimitDisplay data={rateLimitData} onReset={handleReset} />
      )}
    </div>
  );
}
