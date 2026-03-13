import { render, screen, cleanup } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { UsageLimits } from "../UsageLimits";

// Mock the useUsageLimits hook
const mockUseUsageLimits = vi.fn();
vi.mock("../useUsageLimits", () => ({
  useUsageLimits: () => mockUseUsageLimits(),
}));

// Mock Popover to render children directly (Radix portals don't work in happy-dom)
vi.mock("@/components/molecules/Popover/Popover", () => ({
  Popover: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  PopoverTrigger: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  PopoverContent: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

afterEach(() => {
  cleanup();
  mockUseUsageLimits.mockReset();
});

function makeUsage({
  dailyUsed = 500,
  dailyLimit = 10000,
  weeklyUsed = 2000,
  weeklyLimit = 50000,
}: {
  dailyUsed?: number;
  dailyLimit?: number;
  weeklyUsed?: number;
  weeklyLimit?: number;
} = {}) {
  const future = new Date(Date.now() + 3600 * 1000); // 1h from now
  return {
    daily: { used: dailyUsed, limit: dailyLimit, resets_at: future },
    weekly: { used: weeklyUsed, limit: weeklyLimit, resets_at: future },
  };
}

describe("UsageLimits", () => {
  it("renders nothing while loading", () => {
    mockUseUsageLimits.mockReturnValue({ data: undefined, isLoading: true });
    const { container } = render(<UsageLimits />);
    expect(container.innerHTML).toBe("");
  });

  it("renders nothing when no limits are configured", () => {
    mockUseUsageLimits.mockReturnValue({
      data: makeUsage({ dailyLimit: 0, weeklyLimit: 0 }),
      isLoading: false,
    });
    const { container } = render(<UsageLimits />);
    expect(container.innerHTML).toBe("");
  });

  it("renders the usage button when limits exist", () => {
    mockUseUsageLimits.mockReturnValue({
      data: makeUsage(),
      isLoading: false,
    });
    render(<UsageLimits />);
    expect(screen.getByRole("button", { name: /usage limits/i })).toBeDefined();
  });

  it("displays daily and weekly usage percentages", () => {
    mockUseUsageLimits.mockReturnValue({
      data: makeUsage({ dailyUsed: 5000, dailyLimit: 10000 }),
      isLoading: false,
    });
    render(<UsageLimits />);

    expect(screen.getByText("50% used")).toBeDefined();
    expect(screen.getByText("Today")).toBeDefined();
    expect(screen.getByText("This week")).toBeDefined();
    expect(screen.getByText("Usage limits")).toBeDefined();
  });

  it("shows only weekly bar when daily limit is 0", () => {
    mockUseUsageLimits.mockReturnValue({
      data: makeUsage({
        dailyLimit: 0,
        weeklyUsed: 25000,
        weeklyLimit: 50000,
      }),
      isLoading: false,
    });
    render(<UsageLimits />);

    expect(screen.getByText("This week")).toBeDefined();
    expect(screen.queryByText("Today")).toBeNull();
  });

  it("caps percentage at 100% when over limit", () => {
    mockUseUsageLimits.mockReturnValue({
      data: makeUsage({ dailyUsed: 15000, dailyLimit: 10000 }),
      isLoading: false,
    });
    render(<UsageLimits />);

    expect(screen.getByText("100% used")).toBeDefined();
  });

  it("shows learn more link to credits page", () => {
    mockUseUsageLimits.mockReturnValue({
      data: makeUsage(),
      isLoading: false,
    });
    render(<UsageLimits />);

    const link = screen.getByText("Learn more about usage limits");
    expect(link).toBeDefined();
    expect(link.closest("a")?.getAttribute("href")).toBe("/profile/credits");
  });
});
