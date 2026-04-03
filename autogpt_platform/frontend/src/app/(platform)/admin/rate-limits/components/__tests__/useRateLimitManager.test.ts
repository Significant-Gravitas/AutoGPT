import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, cleanup } from "@testing-library/react";

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

const mockGetV2GetUserRateLimit = vi.fn();
const mockGetV2SearchUsersByNameOrEmail = vi.fn();
const mockPostV2ResetUserRateLimitUsage = vi.fn();
const mockPostV2SetUserRateLimitTier = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/admin/admin", () => ({
  getV2GetUserRateLimit: (...args: unknown[]) =>
    mockGetV2GetUserRateLimit(...args),
  getV2SearchUsersByNameOrEmail: (...args: unknown[]) =>
    mockGetV2SearchUsersByNameOrEmail(...args),
  postV2ResetUserRateLimitUsage: (...args: unknown[]) =>
    mockPostV2ResetUserRateLimitUsage(...args),
  postV2SetUserRateLimitTier: (...args: unknown[]) =>
    mockPostV2SetUserRateLimitTier(...args),
}));

import { useRateLimitManager } from "../useRateLimitManager";

function makeRateLimitResponse(overrides = {}) {
  return {
    user_id: "user-123",
    user_email: "alice@example.com",
    daily_token_limit: 10000,
    weekly_token_limit: 50000,
    daily_tokens_used: 2500,
    weekly_tokens_used: 10000,
    tier: "FREE",
    ...overrides,
  };
}

beforeEach(() => {
  mockToast.mockClear();
  mockGetV2GetUserRateLimit.mockReset();
  mockGetV2SearchUsersByNameOrEmail.mockReset();
  mockPostV2ResetUserRateLimitUsage.mockReset();
  mockPostV2SetUserRateLimitTier.mockReset();
});

afterEach(() => {
  cleanup();
});

describe("useRateLimitManager", () => {
  it("returns initial state", () => {
    const { result } = renderHook(() => useRateLimitManager());

    expect(result.current.isSearching).toBe(false);
    expect(result.current.isLoadingRateLimit).toBe(false);
    expect(result.current.searchResults).toEqual([]);
    expect(result.current.selectedUser).toBeNull();
    expect(result.current.rateLimitData).toBeNull();
  });

  it("handleSearch does nothing for empty query", async () => {
    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleSearch("  ");
    });

    expect(mockGetV2GetUserRateLimit).not.toHaveBeenCalled();
    expect(mockGetV2SearchUsersByNameOrEmail).not.toHaveBeenCalled();
  });

  it("handleSearch does direct lookup for email input", async () => {
    const data = makeRateLimitResponse();
    mockGetV2GetUserRateLimit.mockResolvedValue({ status: 200, data });

    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleSearch("alice@example.com");
    });

    expect(mockGetV2GetUserRateLimit).toHaveBeenCalledWith({
      email: "alice@example.com",
    });
    expect(result.current.rateLimitData).toEqual(data);
    expect(result.current.selectedUser).toEqual({
      user_id: "user-123",
      user_email: "alice@example.com",
    });
  });

  it("handleSearch does direct lookup for UUID input", async () => {
    const uuid = "550e8400-e29b-41d4-a716-446655440000";
    const data = makeRateLimitResponse({ user_id: uuid });
    mockGetV2GetUserRateLimit.mockResolvedValue({ status: 200, data });

    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleSearch(uuid);
    });

    expect(mockGetV2GetUserRateLimit).toHaveBeenCalledWith({
      user_id: uuid,
    });
    expect(result.current.rateLimitData).toEqual(data);
  });

  it("handleSearch shows error toast on direct lookup failure", async () => {
    mockGetV2GetUserRateLimit.mockResolvedValue({ status: 404 });

    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleSearch("alice@example.com");
    });

    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Error",
        variant: "destructive",
      }),
    );
    expect(result.current.rateLimitData).toBeNull();
  });

  it("handleSearch does fuzzy search for partial text", async () => {
    const users = [
      { user_id: "u1", user_email: "alice@example.com" },
      { user_id: "u2", user_email: "bob@example.com" },
    ];
    mockGetV2SearchUsersByNameOrEmail.mockResolvedValue({
      status: 200,
      data: users,
    });

    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleSearch("alice");
    });

    expect(mockGetV2SearchUsersByNameOrEmail).toHaveBeenCalledWith({
      query: "alice",
      limit: 20,
    });
    expect(result.current.searchResults).toEqual(users);
  });

  it("handleSearch shows toast when fuzzy search returns no results", async () => {
    mockGetV2SearchUsersByNameOrEmail.mockResolvedValue({
      status: 200,
      data: [],
    });

    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleSearch("nonexistent");
    });

    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "No results" }),
    );
    expect(result.current.searchResults).toEqual([]);
  });

  it("handleSearch shows error toast on fuzzy search failure", async () => {
    mockGetV2SearchUsersByNameOrEmail.mockResolvedValue({ status: 500 });

    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleSearch("alice");
    });

    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Error",
        variant: "destructive",
      }),
    );
  });

  it("handleSelectUser fetches rate limit for selected user", async () => {
    const data = makeRateLimitResponse();
    mockGetV2GetUserRateLimit.mockResolvedValue({ status: 200, data });

    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleSelectUser({
        user_id: "user-123",
        user_email: "alice@example.com",
      });
    });

    expect(mockGetV2GetUserRateLimit).toHaveBeenCalledWith({
      user_id: "user-123",
    });
    expect(result.current.selectedUser).toEqual({
      user_id: "user-123",
      user_email: "alice@example.com",
    });
    expect(result.current.rateLimitData).toEqual(data);
  });

  it("handleSelectUser shows error toast on fetch failure", async () => {
    mockGetV2GetUserRateLimit.mockResolvedValue({ status: 500 });

    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleSelectUser({
        user_id: "user-123",
        user_email: "alice@example.com",
      });
    });

    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Error",
        variant: "destructive",
      }),
    );
    expect(result.current.rateLimitData).toBeNull();
  });

  it("handleReset calls reset endpoint and updates data", async () => {
    const initial = makeRateLimitResponse({ daily_tokens_used: 5000 });
    const after = makeRateLimitResponse({ daily_tokens_used: 0 });
    mockGetV2GetUserRateLimit.mockResolvedValue({ status: 200, data: initial });
    mockPostV2ResetUserRateLimitUsage.mockResolvedValue({
      status: 200,
      data: after,
    });

    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleSelectUser({
        user_id: "user-123",
        user_email: "alice@example.com",
      });
    });

    await act(async () => {
      await result.current.handleReset(false);
    });

    expect(mockPostV2ResetUserRateLimitUsage).toHaveBeenCalledWith({
      user_id: "user-123",
      reset_weekly: false,
    });
    expect(result.current.rateLimitData).toEqual(after);
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Success" }),
    );
  });

  it("handleReset does nothing when no rate limit data", async () => {
    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleReset(false);
    });

    expect(mockPostV2ResetUserRateLimitUsage).not.toHaveBeenCalled();
  });

  it("handleReset shows error toast on failure", async () => {
    const initial = makeRateLimitResponse();
    mockGetV2GetUserRateLimit.mockResolvedValue({ status: 200, data: initial });
    mockPostV2ResetUserRateLimitUsage.mockRejectedValue(
      new Error("network error"),
    );

    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleSelectUser({
        user_id: "user-123",
        user_email: "alice@example.com",
      });
    });

    await act(async () => {
      await result.current.handleReset(true);
    });

    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Error",
        description: "Failed to reset rate limit usage.",
        variant: "destructive",
      }),
    );
  });

  it("handleTierChange calls set tier and re-fetches", async () => {
    const initial = makeRateLimitResponse({ tier: "FREE" });
    const updated = makeRateLimitResponse({ tier: "PRO" });
    mockGetV2GetUserRateLimit
      .mockResolvedValueOnce({ status: 200, data: initial })
      .mockResolvedValueOnce({ status: 200, data: updated });
    mockPostV2SetUserRateLimitTier.mockResolvedValue({ status: 200 });

    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleSelectUser({
        user_id: "user-123",
        user_email: "alice@example.com",
      });
    });

    await act(async () => {
      await result.current.handleTierChange("PRO");
    });

    expect(mockPostV2SetUserRateLimitTier).toHaveBeenCalledWith({
      user_id: "user-123",
      tier: "PRO",
    });
    expect(result.current.rateLimitData).toEqual(updated);
  });

  it("handleTierChange does nothing when no rate limit data", async () => {
    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleTierChange("PRO");
    });

    expect(mockPostV2SetUserRateLimitTier).not.toHaveBeenCalled();
  });

  it("handleReset throws when endpoint returns non-200 status", async () => {
    const initial = makeRateLimitResponse({ daily_tokens_used: 5000 });
    mockGetV2GetUserRateLimit.mockResolvedValue({ status: 200, data: initial });
    mockPostV2ResetUserRateLimitUsage.mockResolvedValue({ status: 500 });

    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleSelectUser({
        user_id: "user-123",
        user_email: "alice@example.com",
      });
    });

    await act(async () => {
      await result.current.handleReset(false);
    });

    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Error",
        description: "Failed to reset rate limit usage.",
        variant: "destructive",
      }),
    );
  });

  it("handleTierChange throws when set-tier endpoint returns non-200", async () => {
    const initial = makeRateLimitResponse({ tier: "FREE" });
    mockGetV2GetUserRateLimit.mockResolvedValue({ status: 200, data: initial });
    mockPostV2SetUserRateLimitTier.mockResolvedValue({ status: 500 });

    const { result } = renderHook(() => useRateLimitManager());

    await act(async () => {
      await result.current.handleSelectUser({
        user_id: "user-123",
        user_email: "alice@example.com",
      });
    });

    await expect(
      act(async () => {
        await result.current.handleTierChange("PRO");
      }),
    ).rejects.toThrow("Failed to update tier");
  });
});
