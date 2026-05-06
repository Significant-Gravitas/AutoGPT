import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BriefingTabContent } from "../BriefingTabContent";

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

afterEach(() => {
  mockUseGetV2GetCopilotUsage.mockReset();
  mockUseGetFlag.mockReset();
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

describe("BriefingTabContent — UsageSection", () => {
  it("renders nothing when usage fetch has not succeeded", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
    });
    mockUseGetFlag.mockReturnValue(false);
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
    render(<BriefingTabContent activeTab="all" agents={[]} />);

    expect(screen.getByText("Usage limits")).toBeDefined();
    expect(screen.getByText("Pro plan")).toBeDefined();
    expect(screen.getByText("12% used")).toBeDefined();
    expect(screen.getByText("4% used")).toBeDefined();
    expect(screen.getByText("Today")).toBeDefined();
    expect(screen.getByText("This week")).toBeDefined();
    expect(screen.getByText("Manage billing")).toBeDefined();
  });

  it("never renders the legacy 'Reset daily limit' button when daily is exhausted", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 100, weeklyPercent: 40 }),
      isSuccess: true,
    });
    mockUseGetFlag.mockReturnValue(true);
    render(<BriefingTabContent activeTab="all" agents={[]} />);
    expect(screen.queryByText(/Reset daily limit/)).toBeNull();
  });

  it("shows 'Manage billing' when billing flag is on, regardless of usage", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 100, weeklyPercent: 40 }),
      isSuccess: true,
    });
    mockUseGetFlag.mockReturnValue(true);
    render(<BriefingTabContent activeTab="all" agents={[]} />);
    expect(screen.getByText("Manage billing")).toBeDefined();
    expect(screen.queryByText("Go to billing")).toBeNull();
  });

  it("hides 'Manage billing' when billing flag is off", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 100, weeklyPercent: 40 }),
      isSuccess: true,
    });
    mockUseGetFlag.mockReturnValue(false);
    render(<BriefingTabContent activeTab="all" agents={[]} />);
    expect(screen.queryByText("Manage billing")).toBeNull();
    expect(screen.queryByText("Go to billing")).toBeNull();
  });

  it("still shows 'Manage billing' when both daily and weekly are exhausted", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 100, weeklyPercent: 100 }),
      isSuccess: true,
    });
    mockUseGetFlag.mockReturnValue(true);
    render(<BriefingTabContent activeTab="all" agents={[]} />);
    expect(screen.getByText("Manage billing")).toBeDefined();
    expect(screen.queryByText(/Reset daily limit/)).toBeNull();
  });

  it("renders <1% used when percent is >0 but rounds to 0", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 0.4, weeklyPercent: 0 }),
      isSuccess: true,
    });
    mockUseGetFlag.mockReturnValue(false);
    render(<BriefingTabContent activeTab="all" agents={[]} />);
    expect(screen.getByText("<1% used")).toBeDefined();
  });

  it("dispatches to ExecutionListSection for running/attention/completed tabs", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
    });
    mockUseGetFlag.mockReturnValue(false);

    for (const tab of ["running", "attention", "completed"] as const) {
      const { unmount } = render(
        <BriefingTabContent activeTab={tab} agents={[]} />,
      );
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

    for (const tab of ["listening", "scheduled", "idle"] as const) {
      const { unmount } = render(
        <BriefingTabContent activeTab={tab} agents={[]} />,
      );
      expect(screen.getByText(/No/i)).toBeDefined();
      unmount();
    }
  });
});
