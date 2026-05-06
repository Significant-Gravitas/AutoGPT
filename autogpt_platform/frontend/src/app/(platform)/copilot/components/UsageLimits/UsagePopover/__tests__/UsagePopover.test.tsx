import { render, screen, cleanup } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { UsagePopover } from "../UsagePopover";

const mockUseUsagePopover = vi.fn();
vi.mock("../useUsagePopover", () => ({
  useUsagePopover: () => mockUseUsagePopover(),
}));

vi.mock("../../StorageBar", () => ({
  StorageBar: () => null,
}));

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
  mockUseUsagePopover.mockReset();
});

beforeEach(() => {
  mockUseUsagePopover.mockReturnValue({ usage: undefined, isSuccess: false });
});

interface UsageOverrides {
  dailyPercent?: number | null;
  weeklyPercent?: number | null;
  tier?: string | null;
}

function makeUsage({
  dailyPercent = 5,
  weeklyPercent = 4,
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

describe("UsagePopover", () => {
  it("renders nothing while loading", () => {
    const { container } = render(<UsagePopover />);
    expect(container.innerHTML).toBe("");
  });

  it("renders nothing when no limits are configured", () => {
    mockUseUsagePopover.mockReturnValue({
      usage: makeUsage({ dailyPercent: null, weeklyPercent: null }),
      isSuccess: true,
    });
    const { container } = render(<UsagePopover />);
    expect(container.innerHTML).toBe("");
  });

  it("renders the trigger button and panel when limits exist", () => {
    mockUseUsagePopover.mockReturnValue({
      usage: makeUsage({ dailyPercent: 50 }),
      isSuccess: true,
    });
    render(<UsagePopover />);

    expect(screen.getByRole("button", { name: /usage limits/i })).toBeDefined();
    expect(screen.getByText("Usage limits")).toBeDefined();
    expect(screen.getByText("Today")).toBeDefined();
    expect(screen.getByText("This week")).toBeDefined();
    expect(screen.getByText("50% used")).toBeDefined();
  });

  it("shows only the weekly bar when daily is null", () => {
    mockUseUsagePopover.mockReturnValue({
      usage: makeUsage({ dailyPercent: null, weeklyPercent: 50 }),
      isSuccess: true,
    });
    render(<UsagePopover />);

    expect(screen.getByText("This week")).toBeDefined();
    expect(screen.queryByText("Today")).toBeNull();
  });

  it("caps the bar width at 100% when over the limit", () => {
    mockUseUsagePopover.mockReturnValue({
      usage: makeUsage({ dailyPercent: 150 }),
      isSuccess: true,
    });
    render(<UsagePopover />);

    const dailyBar = screen.getByRole("progressbar", { name: /today usage/i });
    expect(dailyBar.getAttribute("aria-valuenow")).toBe("100");
  });

  it("renders the tier label", () => {
    mockUseUsagePopover.mockReturnValue({
      usage: makeUsage({ tier: "PRO" }),
      isSuccess: true,
    });
    render(<UsagePopover />);

    expect(screen.getByText("Pro plan")).toBeDefined();
  });

  it("never renders the 'Go to billing' button (handled by the card)", () => {
    mockUseUsagePopover.mockReturnValue({
      usage: makeUsage({ dailyPercent: 100 }),
      isSuccess: true,
    });
    render(<UsagePopover />);

    expect(screen.queryByText("Go to billing")).toBeNull();
  });
});
