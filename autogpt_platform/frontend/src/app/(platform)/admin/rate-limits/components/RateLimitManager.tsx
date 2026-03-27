"use client";

import { AdminUserSearch } from "../../components/AdminUserSearch";
import { RateLimitDisplay } from "./RateLimitDisplay";
import { useRateLimitManager } from "./useRateLimitManager";

export function RateLimitManager() {
  const {
    isSearching,
    isLoadingRateLimit,
    searchResults,
    selectedUser,
    rateLimitData,
    handleSearch,
    handleSelectUser,
    handleReset,
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
        <p className="mt-1.5 text-xs text-gray-500">
          Exact email or user ID does a direct lookup. Partial text searches
          user history.
        </p>
      </div>

      {/* User selection list -- always require explicit selection */}
      {searchResults.length >= 1 && !selectedUser && (
        <div className="rounded-md border bg-white p-4">
          <h3 className="mb-2 text-sm font-medium text-gray-700">
            Select a user ({searchResults.length}{" "}
            {searchResults.length === 1 ? "result" : "results"})
          </h3>
          <ul className="divide-y">
            {searchResults.map((user) => (
              <li key={user.user_id}>
                <button
                  className="w-full px-2 py-2 text-left text-sm hover:bg-gray-100"
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
      {selectedUser && searchResults.length >= 1 && (
        <div className="rounded-md border border-blue-200 bg-blue-50 px-4 py-2 text-sm">
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
