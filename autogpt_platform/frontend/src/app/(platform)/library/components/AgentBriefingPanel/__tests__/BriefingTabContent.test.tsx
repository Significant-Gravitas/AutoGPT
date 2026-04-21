import { render, screen, cleanup } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BriefingTabContent } from "../BriefingTabContent";

const mockUseGetV2GetCopilotUsage = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  useGetV2GetCopilotUsage: (opts: {
    query?: { select?: (r: { data: unknown }) => unknown };
  }) => {
    const ret = mockUseGetV2GetCopilotUsage(opts) as { data?: unknown };
    // Exercise the `select` callback so its line counts as covered.
    if (ret?.data !== undefined && typeof opts?.query?.select === "function") {
      opts.query.select({ data: ret.data });
    }
    return ret;
  },
}));

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

const mockUseCredits = vi.fn();
vi.mock("@/hooks/useCredits", () => ({
  default: (opts: unknown) => mockUseCredits(opts),
}));

const mockResetUsage = vi.fn();
vi.mock("@/app/(platform)/copilot/hooks/useResetRateLimit", () => ({
  useResetRateLimit: () => ({
    resetUsage: mockResetUsage,
    isPending: false,
  }),
}));

afterEach(() => {
  cleanup();
  mockUseGetV2GetCopilotUsage.mockReset();
  mockUseGetFlag.mockReset();
  mockUseCredits.mockReset();
  mockResetUsage.mockReset();
});

function makeUsage({
  dailyPercent = 5,
  weeklyPercent = 4,
  tier = "FREE",
  resetCost = 500,
}: {
  dailyPercent?: number | null;
  weeklyPercent?: number | null;
  tier?: string;
  resetCost?: number;
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
    reset_cost: resetCost,
  };
}

describe("BriefingTabContent — UsageSection", () => {
  it("renders nothing when usage fetch has not succeeded", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
    });
    mockUseGetFlag.mockReturnValue(false);
    mockUseCredits.mockReturnValue({ credits: 1000, fetchCredits: vi.fn() });
    const { container } = render(
      <BriefingTabContent activeTab="all" agents={[]} />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders nothing when both windows are null (no limits configured)", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: null, weeklyPercent: null }),
      isSuccess: true,
    });
    mockUseGetFlag.mockReturnValue(false);
    mockUseCredits.mockReturnValue({ credits: 1000, fetchCredits: vi.fn() });
    const { container } = render(
      <BriefingTabContent activeTab="all" agents={[]} />,
    );
    expect(container.innerHTML).toBe("");
  });

  it("renders tier badge + daily+weekly meters at normal usage", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 12, weeklyPercent: 4, tier: "PRO" }),
      isSuccess: true,
    });
    mockUseGetFlag.mockReturnValue(true);
    mockUseCredits.mockReturnValue({ credits: 1000, fetchCredits: vi.fn() });
    render(<BriefingTabContent activeTab="all" agents={[]} />);

    expect(screen.getByText("Usage limits")).toBeDefined();
    expect(screen.getByText("Pro plan")).toBeDefined();
    expect(screen.getByText("12% used")).toBeDefined();
    expect(screen.getByText("4% used")).toBeDefined();
    expect(screen.getByText("Today")).toBeDefined();
    expect(screen.getByText("This week")).toBeDefined();
    expect(screen.getByText("Manage billing")).toBeDefined();
  });

  it("shows reset button when daily limit is exhausted and user has credits", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 100, weeklyPercent: 40, resetCost: 500 }),
      isSuccess: true,
    });
    mockUseGetFlag.mockReturnValue(true);
    mockUseCredits.mockReturnValue({ credits: 1000, fetchCredits: vi.fn() });
    render(<BriefingTabContent activeTab="all" agents={[]} />);

    expect(screen.getByText(/Reset daily limit/)).toBeDefined();
  });

  it("shows 'Add credits' CTA when daily exhausted but user lacks credits", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 100, weeklyPercent: 40, resetCost: 500 }),
      isSuccess: true,
    });
    mockUseGetFlag.mockReturnValue(true);
    mockUseCredits.mockReturnValue({ credits: 10, fetchCredits: vi.fn() });
    render(<BriefingTabContent activeTab="all" agents={[]} />);

    expect(screen.getByText("Add credits to reset")).toBeDefined();
    expect(screen.queryByText(/Reset daily limit/)).toBeNull();
  });

  it("hides reset CTAs when the weekly limit is also exhausted", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({
        dailyPercent: 100,
        weeklyPercent: 100,
        resetCost: 500,
      }),
      isSuccess: true,
    });
    mockUseGetFlag.mockReturnValue(true);
    mockUseCredits.mockReturnValue({ credits: 1000, fetchCredits: vi.fn() });
    render(<BriefingTabContent activeTab="all" agents={[]} />);

    expect(screen.queryByText(/Reset daily limit/)).toBeNull();
    expect(screen.queryByText("Add credits to reset")).toBeNull();
  });

  it("renders <1% used when percent is >0 but rounds to 0", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 0.4, weeklyPercent: 0 }),
      isSuccess: true,
    });
    mockUseGetFlag.mockReturnValue(false);
    mockUseCredits.mockReturnValue({ credits: 1000, fetchCredits: vi.fn() });
    render(<BriefingTabContent activeTab="all" agents={[]} />);

    expect(screen.getByText("<1% used")).toBeDefined();
  });

  it("dispatches to ExecutionListSection for running/attention/completed tabs", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
    });
    mockUseGetFlag.mockReturnValue(false);
    mockUseCredits.mockReturnValue({ credits: 1000, fetchCredits: vi.fn() });

    for (const tab of ["running", "attention", "completed"] as const) {
      const { unmount } = render(
        <BriefingTabContent activeTab={tab} agents={[]} />,
      );
      // Empty list -> EmptyMessage renders for each of the execution tabs.
      expect(
        screen.getByText(/No agents|No recently completed/i),
      ).toBeDefined();
      unmount();
    }
  });

  it("dispatches to AgentListSection for listening/scheduled/idle tabs", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
    });
    mockUseGetFlag.mockReturnValue(false);
    mockUseCredits.mockReturnValue({ credits: 1000, fetchCredits: vi.fn() });

    for (const tab of ["listening", "scheduled", "idle"] as const) {
      const { unmount } = render(
        <BriefingTabContent activeTab={tab} agents={[]} />,
      );
      expect(screen.getByText(/No/i)).toBeDefined();
      unmount();
    }
  });
});
