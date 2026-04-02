import { render, screen, cleanup } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { UsageLimits } from "../UsageLimits";

// Mock the generated Orval hook
const mockUseGetV2GetCopilotUsage = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useGetV2GetCopilotUsage: (opts: unknown) => mockUseGetV2GetCopilotUsage(opts),
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
  mockUseGetV2GetCopilotUsage.mockReset();
});

function makeUsage({
  dailyUsed = 500,
  dailyLimit = 10000,
  weeklyUsed = 2000,
  weeklyLimit = 50000,
  tier = "FREE",
}: {
  dailyUsed?: number;
  dailyLimit?: number;
  weeklyUsed?: number;
  weeklyLimit?: number;
  tier?: string;
} = {}) {
  const future = new Date(Date.now() + 3600 * 1000); // 1h from now
  return {
    daily: { used: dailyUsed, limit: dailyLimit, resets_at: future },
    weekly: { used: weeklyUsed, limit: weeklyLimit, resets_at: future },
    tier,
  };
}

describe("UsageLimits", () => {
  it("renders nothing while loading", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isLoading: true,
    });
    const { container } = render(<UsageLimits />);
    expect(container.innerHTML).toBe("");
  });

  it("renders nothing when no limits are configured", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyLimit: 0, weeklyLimit: 0 }),
      isLoading: false,
    });
    const { container } = render(<UsageLimits />);
    expect(container.innerHTML).toBe("");
  });

  it("renders the usage button when limits exist", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage(),
      isLoading: false,
    });
    render(<UsageLimits />);
    expect(screen.getByRole("button", { name: /usage limits/i })).toBeDefined();
  });

  it("displays daily and weekly usage percentages", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
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
    mockUseGetV2GetCopilotUsage.mockReturnValue({
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
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyUsed: 15000, dailyLimit: 10000 }),
      isLoading: false,
    });
    render(<UsageLimits />);

    expect(screen.getByText("100% used")).toBeDefined();
  });

  it("displays the user tier label", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ tier: "PRO" }),
      isLoading: false,
    });
    render(<UsageLimits />);

    expect(screen.getByText("Pro plan")).toBeDefined();
  });

  it("shows learn more link to credits page", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage(),
      isLoading: false,
    });
    render(<UsageLimits />);

    const link = screen.getByText("Learn more about usage limits");
    expect(link).toBeDefined();
    expect(link.closest("a")?.getAttribute("href")).toBe("/profile/credits");
  });
});
