"use client";

import { AdminUserSearch } from "../../components/AdminUserSearch";
import { RateLimitDisplay } from "./RateLimitDisplay";
import { useRateLimitManager } from "./useRateLimitManager";
import { Text } from "@/components/atoms/Text/Text";

export function RateLimitManager() {
  const {
    isSearching,
    isLoadingRateLimit,
    searchResults,
    selectedUser,
    rateLimitData,
    tierMultipliers,
    handleSearch,
    handleSelectUser,
    handleReset,
    handleTierChange,
  } = useRateLimitManager();

  return (
    <div className="space-y-6">
      <div className="rounded-md border bg-white p-6">
        <label className="mb-2 block text-sm font-medium">Search User</label>
        <AdminUserSearch
          onSearch={handleSearch}
          placeholder="Search by name, email, or user ID..."
          isLoading={isSearching}
        />
        <Text variant="small" tone="muted" className="mt-1.5">
          Exact email or user ID does a direct lookup. Partial text searches
          user history.
        </Text>
      </div>

      {/* User selection list -- always require explicit selection */}
      {searchResults.length >= 1 && !selectedUser && (
        <div className="rounded-md border bg-white p-4">
          <Text variant="body-medium" as="h3" tone="secondary" className="mb-2">
            Select a user ({searchResults.length}{" "}
            {searchResults.length === 1 ? "result" : "results"})
          </Text>
          <ul className="divide-y">
            {searchResults.map((user) => (
              <li key={user.user_id}>
                <button
                  className="w-full px-2 py-2 text-left text-sm hover:bg-zinc-100"
                  onClick={() => handleSelectUser(user)}
                >
                  <span className="font-medium">{user.user_email}</span>
                  <span className="ml-2 text-xs text-zinc-500">
                    {user.user_id}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Show selected user */}
      {selectedUser && searchResults.length >= 1 && (
        <div className="rounded-md border border-blue-200 bg-blue-50 px-4 py-2 text-sm">
          Selected:{" "}
          <span className="font-medium">{selectedUser.user_email}</span>
          <span className="ml-2 text-xs text-zinc-500">
            {selectedUser.user_id}
          </span>
        </div>
      )}

      {isLoadingRateLimit && (
        <div className="py-4 text-center text-sm text-zinc-500">
          Loading rate limits...
        </div>
      )}

      {rateLimitData && (
        <RateLimitDisplay
          data={rateLimitData}
          onReset={handleReset}
          onTierChange={handleTierChange}
          tierMultipliers={tierMultipliers}
        />
      )}
    </div>
  );
}
