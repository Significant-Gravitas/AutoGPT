import { render, screen, cleanup } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { UsageLimits } from "../UsageLimits";

// Mock the generated Orval hook, exercising the `select` callback so its
// line counts as covered alongside the rest of the options.
const mockUseGetV2GetCopilotUsage = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useGetV2GetCopilotUsage: (opts: {
    query?: { select?: (r: { data: unknown }) => unknown };
  }) => {
    const ret = mockUseGetV2GetCopilotUsage(opts) as { data?: unknown };
    if (ret?.data !== undefined && typeof opts?.query?.select === "function") {
      opts.query.select({ data: ret.data });
    }
    return ret;
  },
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
  dailyPercent = 5,
  weeklyPercent = 4,
  tier = "BASIC",
}: {
  dailyPercent?: number | null;
  weeklyPercent?: number | null;
  tier?: string;
} = {}) {
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

describe("UsageLimits", () => {
  it("renders nothing while loading", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
    });
    const { container } = render(<UsageLimits />);
    expect(container.innerHTML).toBe("");
  });

  it("renders nothing when no limits are configured", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: null, weeklyPercent: null }),
      isSuccess: true,
    });
    const { container } = render(<UsageLimits />);
    expect(container.innerHTML).toBe("");
  });

  it("renders the usage button when limits exist", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage(),
      isSuccess: true,
    });
    render(<UsageLimits />);
    expect(screen.getByRole("button", { name: /usage limits/i })).toBeDefined();
  });

  it("displays daily and weekly percentage", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 50, weeklyPercent: 4 }),
      isSuccess: true,
    });
    render(<UsageLimits />);

    expect(screen.getByText("50% used")).toBeDefined();
    expect(screen.getByText("Today")).toBeDefined();
    expect(screen.getByText("This week")).toBeDefined();
    expect(screen.getByText("Usage limits")).toBeDefined();
  });

  it("shows only weekly bar when daily is null", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: null, weeklyPercent: 50 }),
      isSuccess: true,
    });
    render(<UsageLimits />);

    expect(screen.getByText("This week")).toBeDefined();
    expect(screen.queryByText("Today")).toBeNull();
  });

  it("caps bar width at 100% when over limit", () => {
    // 150% exercises the clamp — 100% exactly is merely exhausted, not over.
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 150 }),
      isSuccess: true,
    });
    render(<UsageLimits />);

    const dailyBar = screen.getByRole("progressbar", { name: /today usage/i });
    expect(dailyBar.getAttribute("aria-valuenow")).toBe("100");
  });

  it("displays the user tier label", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ tier: "PRO" }),
      isSuccess: true,
    });
    render(<UsageLimits />);

    expect(screen.getByText("Pro plan")).toBeDefined();
  });

  it("shows learn more link to credits page", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage(),
      isSuccess: true,
    });
    render(<UsageLimits />);

    const link = screen.getByText("Learn more about usage limits");
    expect(link).toBeDefined();
    expect(link.closest("a")?.getAttribute("href")).toBe("/profile/credits");
  });
});
