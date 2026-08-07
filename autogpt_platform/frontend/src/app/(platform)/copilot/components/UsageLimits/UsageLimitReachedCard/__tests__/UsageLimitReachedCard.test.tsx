import { http, HttpResponse, type JsonBodyType } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { UsageLimitReachedCard } from "../UsageLimitReachedCard";

const mockUseGetFlag = vi.fn();
vi.mock("@/services/feature-flags/use-get-flag", async () => {
  const actual = await vi.importActual<
    typeof import("@/services/feature-flags/use-get-flag")
  >("@/services/feature-flags/use-get-flag");
  return {
    ...actual,
    useGetFlag: (flag: unknown) => mockUseGetFlag(flag),
  };
});

vi.mock("../../StorageBar", () => ({
  StorageBar: () => null,
}));

afterEach(() => {
  mockUseGetFlag.mockReset();
});

beforeEach(() => {
  mockUseGetFlag.mockReturnValue(true);
});

interface UsageOverrides {
  dailyPercent?: number | null;
  weeklyPercent?: number | null;
  tier?: string | null;
}

function makeUsage({
  dailyPercent = 100,
  weeklyPercent = 40,
  tier = "BASIC",
}: UsageOverrides = {}) {
  const future = new Date(Date.now() + 3600 * 1000).toISOString();
  return {
    daily:
      dailyPercent === null
        ? null
        : { percent_used: dailyPercent, resets_at: future },
    weekly:
      weeklyPercent === null
        ? null
        : { percent_used: weeklyPercent, resets_at: future },
    tier,
  };
}

function mockUsageResponse(body: JsonBodyType) {
  server.use(http.get("*/api/chat/usage", () => HttpResponse.json(body)));
}

describe("UsageLimitReachedCard", () => {
  it("renders nothing on the first paint while data is loading", () => {
    mockUsageResponse(makeUsage());
    const { container } = render(<UsageLimitReachedCard />);
    expect(container.innerHTML).toBe("");
  });

  it("renders nothing when neither daily nor weekly is exhausted", async () => {
    mockUsageResponse(makeUsage({ dailyPercent: 5, weeklyPercent: 4 }));
    const { container } = render(<UsageLimitReachedCard />);
    await waitFor(() => expect(container.innerHTML).toBe(""));
  });

  it("renders the alert with daily and weekly bars when the daily limit is reached", async () => {
    mockUsageResponse(makeUsage());
    render(<UsageLimitReachedCard />);

    expect(await screen.findByRole("alert")).toBeDefined();
    expect(screen.getByText("Usage limit reached")).toBeDefined();
    expect(screen.getByText("Today")).toBeDefined();
    expect(screen.getByText("This week")).toBeDefined();
  });

  it("always shows the 'Go to billing' button when billing is enabled", async () => {
    mockUsageResponse(makeUsage());
    render(<UsageLimitReachedCard />);

    const link = (await screen.findByText("Go to billing")).closest("a");
    expect(link).not.toBeNull();
    expect(link?.getAttribute("href")).toBe("/settings/billing");
  });

  it("hides the 'Go to billing' button when billing is disabled at the platform level", async () => {
    mockUseGetFlag.mockReturnValue(false);
    mockUsageResponse(makeUsage());
    render(<UsageLimitReachedCard />);

    expect(await screen.findByRole("alert")).toBeDefined();
    expect(screen.queryByText("Go to billing")).toBeNull();
  });

  it("renders the tier badge when a tier is set", async () => {
    mockUsageResponse(makeUsage({ tier: "PRO" }));
    render(<UsageLimitReachedCard />);

    expect(await screen.findByText("Pro plan")).toBeDefined();
  });

  it("never renders the legacy 'Reset daily limit' control", async () => {
    mockUsageResponse(makeUsage());
    render(<UsageLimitReachedCard />);

    expect(await screen.findByRole("alert")).toBeDefined();
    expect(screen.queryByText(/Reset daily limit/i)).toBeNull();
  });
});
