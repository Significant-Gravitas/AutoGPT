import { render, screen, cleanup } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { UsageLimitReachedCard } from "../UsageLimitReachedCard";

const mockUseUsageLimitReachedCard = vi.fn();
vi.mock("../useUsageLimitReachedCard", () => ({
  useUsageLimitReachedCard: () => mockUseUsageLimitReachedCard(),
}));

vi.mock("../../StorageBar", () => ({
  StorageBar: () => null,
}));

afterEach(() => {
  cleanup();
  mockUseUsageLimitReachedCard.mockReset();
});

beforeEach(() => {
  mockUseUsageLimitReachedCard.mockReturnValue({
    usage: undefined,
    isSuccess: false,
    isBillingEnabled: true,
  });
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

describe("UsageLimitReachedCard", () => {
  it("renders nothing while usage data is loading", () => {
    const { container } = render(<UsageLimitReachedCard />);
    expect(container.innerHTML).toBe("");
  });

  it("renders the alert with daily and weekly bars when data is ready", () => {
    mockUseUsageLimitReachedCard.mockReturnValue({
      usage: makeUsage(),
      isSuccess: true,
      isBillingEnabled: true,
    });
    render(<UsageLimitReachedCard />);

    expect(screen.getByRole("alert")).toBeDefined();
    expect(screen.getByText("Usage limit reached")).toBeDefined();
    expect(screen.getByText("Today")).toBeDefined();
    expect(screen.getByText("This week")).toBeDefined();
  });

  it("always shows the 'Go to billing' button when billing is enabled", () => {
    mockUseUsageLimitReachedCard.mockReturnValue({
      usage: makeUsage(),
      isSuccess: true,
      isBillingEnabled: true,
    });
    render(<UsageLimitReachedCard />);

    const link = screen.getByText("Go to billing").closest("a");
    expect(link).not.toBeNull();
    expect(link?.getAttribute("href")).toBe("/settings/billing");
  });

  it("hides the 'Go to billing' button when billing is disabled at the platform level", () => {
    mockUseUsageLimitReachedCard.mockReturnValue({
      usage: makeUsage(),
      isSuccess: true,
      isBillingEnabled: false,
    });
    render(<UsageLimitReachedCard />);

    expect(screen.queryByText("Go to billing")).toBeNull();
  });

  it("renders the tier badge when a tier is set", () => {
    mockUseUsageLimitReachedCard.mockReturnValue({
      usage: makeUsage({ tier: "PRO" }),
      isSuccess: true,
      isBillingEnabled: true,
    });
    render(<UsageLimitReachedCard />);

    expect(screen.getByText("Pro")).toBeDefined();
  });

  it("never renders the legacy 'Reset daily limit' control", () => {
    mockUseUsageLimitReachedCard.mockReturnValue({
      usage: makeUsage(),
      isSuccess: true,
      isBillingEnabled: true,
    });
    render(<UsageLimitReachedCard />);

    expect(screen.queryByText(/Reset daily limit/i)).toBeNull();
  });
});
