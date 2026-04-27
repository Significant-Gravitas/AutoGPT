import {
  render,
  screen,
  fireEvent,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RateLimitManager } from "../RateLimitManager";
import type { UserRateLimitResponse } from "@/app/api/__generated__/models/userRateLimitResponse";

const mockHandleSearch = vi.fn();
const mockHandleSelectUser = vi.fn();
const mockHandleReset = vi.fn();
const mockHandleTierChange = vi.fn();

vi.mock("../useRateLimitManager", () => ({
  useRateLimitManager: () => mockHookReturn,
}));

vi.mock("../../../components/AdminUserSearch", () => ({
  AdminUserSearch: ({
    onSearch,
    placeholder,
    isLoading,
  }: {
    onSearch: (q: string) => void;
    placeholder: string;
    isLoading: boolean;
  }) => (
    <div data-testid="admin-user-search">
      <input
        data-testid="search-input"
        placeholder={placeholder}
        disabled={isLoading}
        onKeyDown={(e) => {
          if (e.key === "Enter") onSearch((e.target as HTMLInputElement).value);
        }}
      />
    </div>
  ),
}));

vi.mock("../RateLimitDisplay", () => ({
  RateLimitDisplay: ({
    data,
    onReset,
    onTierChange,
  }: {
    data: UserRateLimitResponse;
    onReset: (rw: boolean) => void;
    onTierChange: (t: string) => void;
  }) => (
    <div data-testid="rate-limit-display">
      <span>{data.user_email ?? data.user_id}</span>
      <button onClick={() => onReset(false)}>mock-reset</button>
      <button onClick={() => onTierChange("PRO")}>mock-tier</button>
    </div>
  ),
}));

let mockHookReturn = buildHookReturn();

function buildHookReturn(overrides: Record<string, unknown> = {}) {
  return {
    isSearching: false,
    isLoadingRateLimit: false,
    searchResults: [] as Array<{ user_id: string; user_email: string }>,
    selectedUser: null as { user_id: string; user_email: string } | null,
    rateLimitData: null as UserRateLimitResponse | null,
    handleSearch: mockHandleSearch,
    handleSelectUser: mockHandleSelectUser,
    handleReset: mockHandleReset,
    handleTierChange: mockHandleTierChange,
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
  mockHandleSearch.mockClear();
  mockHandleSelectUser.mockClear();
  mockHandleReset.mockClear();
  mockHandleTierChange.mockClear();
  mockHookReturn = buildHookReturn();
});

describe("RateLimitManager", () => {
  it("renders the search section", () => {
    render(<RateLimitManager />);
    expect(screen.getByText("Search User")).toBeDefined();
    expect(screen.getByTestId("admin-user-search")).toBeDefined();
  });

  it("renders description text for search", () => {
    render(<RateLimitManager />);
    expect(
      screen.getByText(/Exact email or user ID does a direct lookup/),
    ).toBeDefined();
  });

  it("does not show user list when searchResults is empty", () => {
    render(<RateLimitManager />);
    expect(screen.queryByText(/Select a user/)).toBeNull();
  });

  it("shows user selection list when results exist and no user selected", () => {
    mockHookReturn = buildHookReturn({
      searchResults: [
        { user_id: "u1", user_email: "alice@example.com" },
        { user_id: "u2", user_email: "bob@example.com" },
      ],
    });

    render(<RateLimitManager />);

    expect(screen.getByText("Select a user (2 results)")).toBeDefined();
    expect(screen.getByText("alice@example.com")).toBeDefined();
    expect(screen.getByText("bob@example.com")).toBeDefined();
  });

  it("shows singular 'result' text for single result", () => {
    mockHookReturn = buildHookReturn({
      searchResults: [{ user_id: "u1", user_email: "alice@example.com" }],
    });

    render(<RateLimitManager />);
    expect(screen.getByText("Select a user (1 result)")).toBeDefined();
  });

  it("calls handleSelectUser when a user in the list is clicked", () => {
    const users = [
      { user_id: "u1", user_email: "alice@example.com" },
      { user_id: "u2", user_email: "bob@example.com" },
    ];
    mockHookReturn = buildHookReturn({ searchResults: users });

    render(<RateLimitManager />);

    fireEvent.click(screen.getByText("bob@example.com"));
    expect(mockHandleSelectUser).toHaveBeenCalledWith(users[1]);
  });

  it("hides selection list when a user is selected", () => {
    const users = [{ user_id: "u1", user_email: "alice@example.com" }];
    mockHookReturn = buildHookReturn({
      searchResults: users,
      selectedUser: users[0],
    });

    render(<RateLimitManager />);
    expect(screen.queryByText(/Select a user/)).toBeNull();
  });

  it("shows selected user indicator", () => {
    const users = [{ user_id: "u1", user_email: "alice@example.com" }];
    mockHookReturn = buildHookReturn({
      searchResults: users,
      selectedUser: users[0],
    });

    render(<RateLimitManager />);
    expect(screen.getByText("Selected:")).toBeDefined();
  });

  it("shows loading message when isLoadingRateLimit is true", () => {
    mockHookReturn = buildHookReturn({ isLoadingRateLimit: true });

    render(<RateLimitManager />);
    expect(screen.getByText("Loading rate limits...")).toBeDefined();
  });

  it("renders RateLimitDisplay when rateLimitData is present", () => {
    mockHookReturn = buildHookReturn({
      rateLimitData: {
        user_id: "user-123",
        user_email: "alice@example.com",
        daily_cost_limit_microdollars: 10_000_000,
        weekly_cost_limit_microdollars: 50_000_000,
        daily_cost_used_microdollars: 2_500_000,
        weekly_cost_used_microdollars: 10_000_000,
        tier: "BASIC",
      },
    });

    render(<RateLimitManager />);
    expect(screen.getByTestId("rate-limit-display")).toBeDefined();
    expect(screen.getByText("alice@example.com")).toBeDefined();
  });

  it("does not render RateLimitDisplay when rateLimitData is null", () => {
    render(<RateLimitManager />);
    expect(screen.queryByTestId("rate-limit-display")).toBeNull();
  });

  it("passes handleReset and handleTierChange to RateLimitDisplay", () => {
    mockHookReturn = buildHookReturn({
      rateLimitData: {
        user_id: "user-123",
        user_email: "alice@example.com",
        daily_cost_limit_microdollars: 10_000_000,
        weekly_cost_limit_microdollars: 50_000_000,
        daily_cost_used_microdollars: 2_500_000,
        weekly_cost_used_microdollars: 10_000_000,
        tier: "BASIC",
      },
    });

    render(<RateLimitManager />);

    fireEvent.click(screen.getByText("mock-reset"));
    expect(mockHandleReset).toHaveBeenCalledWith(false);

    fireEvent.click(screen.getByText("mock-tier"));
    expect(mockHandleTierChange).toHaveBeenCalledWith("PRO");
  });
});
