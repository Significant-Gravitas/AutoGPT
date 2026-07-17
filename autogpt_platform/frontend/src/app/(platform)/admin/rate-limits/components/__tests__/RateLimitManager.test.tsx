import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import {
  getGetV2GetUserRateLimitMockHandler,
  getGetV2GetUserRateLimitMockHandler200,
  getGetV2SearchUsersByNameOrEmailMockHandler200,
  getPostV2ResetUserRateLimitUsageMockHandler200,
  getPostV2SetUserRateLimitTierMockHandler200,
} from "@/app/api/__generated__/endpoints/admin/admin.msw";
import { http, HttpResponse } from "msw";
import { RateLimitManager } from "../RateLimitManager";
import type { UserRateLimitResponse } from "@/app/api/__generated__/models/userRateLimitResponse";

const toastSpy = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: toastSpy }),
  toast: toastSpy,
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
    onReset: (rw: boolean) => Promise<void>;
    onTierChange: (t: string) => Promise<void>;
  }) => (
    <div data-testid="rate-limit-display">
      <span>{data.user_email ?? data.user_id}</span>
      <span data-testid="display-tier">{data.tier}</span>
      <span data-testid="display-daily-used">
        {data.daily_cost_used_microdollars}
      </span>
      <button
        onClick={() => {
          onReset(false).catch(() => {});
        }}
      >
        mock-reset
      </button>
      <button
        onClick={() => {
          onTierChange("PRO").catch(() => {});
        }}
      >
        mock-tier
      </button>
    </div>
  ),
}));

const RATE_LIMIT_URL =
  "http://localhost:3000/api/proxy/api/copilot/admin/rate_limit";
const SEARCH_URL =
  "http://localhost:3000/api/proxy/api/copilot/admin/rate_limit/search_users";
const RESET_URL =
  "http://localhost:3000/api/proxy/api/copilot/admin/rate_limit/reset";
const TIER_URL =
  "http://localhost:3000/api/proxy/api/copilot/admin/rate_limit/tier";

function makeRateLimit(
  overrides: Partial<UserRateLimitResponse> = {},
): UserRateLimitResponse {
  return {
    user_id: "user-123",
    user_email: "alice@example.com",
    daily_cost_limit_microdollars: 10_000_000,
    weekly_cost_limit_microdollars: 50_000_000,
    daily_cost_used_microdollars: 2_500_000,
    weekly_cost_used_microdollars: 10_000_000,
    tier: "BASIC",
    ...overrides,
  };
}

function typeAndSearch(query: string) {
  const input = screen.getByTestId("search-input") as HTMLInputElement;
  fireEvent.keyDown(input, { key: "Enter", target: { value: query } });
}

beforeEach(() => {
  toastSpy.mockClear();
});

afterEach(() => {
  cleanup();
});

describe("RateLimitManager - rendering", () => {
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

  it("does not show user list initially", () => {
    render(<RateLimitManager />);
    expect(screen.queryByText(/Select a user/)).toBeNull();
  });

  it("does not render RateLimitDisplay initially", () => {
    render(<RateLimitManager />);
    expect(screen.queryByTestId("rate-limit-display")).toBeNull();
  });
});

describe("RateLimitManager - search behavior", () => {
  it("performs a direct lookup for an email and shows the rate limit display", async () => {
    const data = makeRateLimit();
    server.use(getGetV2GetUserRateLimitMockHandler200(data));

    render(<RateLimitManager />);
    typeAndSearch("alice@example.com");

    expect(await screen.findByTestId("rate-limit-display")).toBeDefined();
    expect(screen.getByText("alice@example.com")).toBeDefined();
  });

  it("performs a direct lookup for a UUID and shows the rate limit display", async () => {
    const uuid = "550e8400-e29b-41d4-a716-446655440000";
    const data = makeRateLimit({ user_id: uuid });
    server.use(getGetV2GetUserRateLimitMockHandler200(data));

    render(<RateLimitManager />);
    typeAndSearch(uuid);

    expect(await screen.findByTestId("rate-limit-display")).toBeDefined();
  });

  it("shows error toast on direct lookup failure", async () => {
    server.use(
      http.get(RATE_LIMIT_URL, () =>
        HttpResponse.json({ detail: "not found" }, { status: 404 }),
      ),
    );

    render(<RateLimitManager />);
    typeAndSearch("alice@example.com");

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Error",
          variant: "destructive",
        }),
      );
    });
    expect(screen.queryByTestId("rate-limit-display")).toBeNull();
  });

  it("does fuzzy search for partial text and renders the result list", async () => {
    server.use(
      getGetV2SearchUsersByNameOrEmailMockHandler200([
        { user_id: "u1", user_email: "alice@example.com" },
        { user_id: "u2", user_email: "bob@example.com" },
      ]),
    );

    render(<RateLimitManager />);
    typeAndSearch("alice");

    expect(await screen.findByText("Select a user (2 results)")).toBeDefined();
    expect(screen.getByText("alice@example.com")).toBeDefined();
    expect(screen.getByText("bob@example.com")).toBeDefined();
  });

  it("shows toast when fuzzy search returns no results", async () => {
    server.use(getGetV2SearchUsersByNameOrEmailMockHandler200([]));

    render(<RateLimitManager />);
    typeAndSearch("nonexistent");

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({ title: "No results" }),
      );
    });
  });

  it("shows error toast on fuzzy search failure", async () => {
    server.use(
      http.get(SEARCH_URL, () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    render(<RateLimitManager />);
    typeAndSearch("alice");

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Error",
          variant: "destructive",
        }),
      );
    });
  });
});

describe("RateLimitManager - user selection list", () => {
  it("shows singular 'result' text for single result", async () => {
    server.use(
      getGetV2SearchUsersByNameOrEmailMockHandler200([
        { user_id: "u1", user_email: "alice@example.com" },
      ]),
    );

    render(<RateLimitManager />);
    typeAndSearch("alice");

    expect(await screen.findByText("Select a user (1 result)")).toBeDefined();
  });

  it("loads rate limit display when a user in the list is clicked", async () => {
    const users = [
      { user_id: "u1", user_email: "alice@example.com" },
      { user_id: "u2", user_email: "bob@example.com" },
    ];
    server.use(
      getGetV2SearchUsersByNameOrEmailMockHandler200(users),
      getGetV2GetUserRateLimitMockHandler200(
        makeRateLimit({ user_id: "u2", user_email: "bob@example.com" }),
      ),
    );

    render(<RateLimitManager />);
    typeAndSearch("user");

    fireEvent.click(await screen.findByText("bob@example.com"));

    expect(await screen.findByTestId("rate-limit-display")).toBeDefined();
    expect(screen.getByText("Selected:")).toBeDefined();
  });

  it("shows error toast when fetching rate limit for selected user fails", async () => {
    server.use(
      getGetV2SearchUsersByNameOrEmailMockHandler200([
        { user_id: "u1", user_email: "alice@example.com" },
      ]),
      http.get(RATE_LIMIT_URL, () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    render(<RateLimitManager />);
    typeAndSearch("alice");

    fireEvent.click(await screen.findByText("alice@example.com"));

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Error",
          variant: "destructive",
        }),
      );
    });
    expect(screen.queryByTestId("rate-limit-display")).toBeNull();
  });
});

describe("RateLimitManager - reset and tier change", () => {
  it("calls reset endpoint, updates rate limit data, and shows success toast", async () => {
    const initial = makeRateLimit({ daily_cost_used_microdollars: 5_000_000 });
    const after = makeRateLimit({ daily_cost_used_microdollars: 0 });

    server.use(
      getGetV2GetUserRateLimitMockHandler200(initial),
      getPostV2ResetUserRateLimitUsageMockHandler200(after),
    );

    render(<RateLimitManager />);
    typeAndSearch("alice@example.com");

    await screen.findByTestId("rate-limit-display");
    fireEvent.click(screen.getByText("mock-reset"));

    await waitFor(() => {
      expect(screen.getByTestId("display-daily-used").textContent).toBe("0");
    });
    expect(toastSpy).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Success" }),
    );
  });

  it("shows error toast when reset endpoint fails", async () => {
    server.use(
      getGetV2GetUserRateLimitMockHandler200(makeRateLimit()),
      http.post(RESET_URL, () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    render(<RateLimitManager />);
    typeAndSearch("alice@example.com");

    await screen.findByTestId("rate-limit-display");
    fireEvent.click(screen.getByText("mock-reset"));

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Error",
          description: "Failed to reset rate limit usage.",
          variant: "destructive",
        }),
      );
    });
  });

  it("changes tier and re-fetches the rate limit data", async () => {
    let callCount = 0;
    server.use(
      http.get(RATE_LIMIT_URL, () => {
        callCount += 1;
        const tier = callCount === 1 ? "BASIC" : "PRO";
        return HttpResponse.json(makeRateLimit({ tier }), { status: 200 });
      }),
      getPostV2SetUserRateLimitTierMockHandler200(),
    );

    render(<RateLimitManager />);
    typeAndSearch("alice@example.com");

    await waitFor(() =>
      expect(screen.getByTestId("display-tier").textContent).toBe("BASIC"),
    );

    fireEvent.click(screen.getByText("mock-tier"));

    await waitFor(() =>
      expect(screen.getByTestId("display-tier").textContent).toBe("PRO"),
    );
  });

  it("does not update tier when the set-tier endpoint fails", async () => {
    let getCount = 0;
    server.use(
      http.get(RATE_LIMIT_URL, () => {
        getCount += 1;
        return HttpResponse.json(makeRateLimit({ tier: "BASIC" }), {
          status: 200,
        });
      }),
      http.post(TIER_URL, () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    render(<RateLimitManager />);
    typeAndSearch("alice@example.com");

    await screen.findByTestId("rate-limit-display");
    expect(getCount).toBe(1);

    fireEvent.click(screen.getByText("mock-tier"));

    // The hook throws on non-200, so it never re-fetches. The displayed tier
    // should remain BASIC and no second GET should fire.
    await waitFor(() => {
      expect(screen.getByTestId("display-tier").textContent).toBe("BASIC");
    });
    expect(getCount).toBe(1);
  });

  it("does not render RateLimitDisplay when search returns no rate limit data", () => {
    // Default initial render: ensure the display is gated on rateLimitData.
    render(<RateLimitManager />);
    expect(screen.queryByTestId("rate-limit-display")).toBeNull();
  });
});

describe("RateLimitManager - loading states", () => {
  it("shows loading message while fetching rate limit after selecting user", async () => {
    server.use(
      getGetV2SearchUsersByNameOrEmailMockHandler200([
        { user_id: "u1", user_email: "alice@example.com" },
      ]),
      // Delayed handler so we can observe the loading state
      getGetV2GetUserRateLimitMockHandler(
        async () =>
          new Promise<UserRateLimitResponse>((resolve) =>
            setTimeout(() => resolve(makeRateLimit()), 50),
          ),
      ),
    );

    render(<RateLimitManager />);
    typeAndSearch("alice");

    fireEvent.click(await screen.findByText("alice@example.com"));

    expect(await screen.findByText("Loading rate limits...")).toBeDefined();
    expect(await screen.findByTestId("rate-limit-display")).toBeDefined();
  });
});
