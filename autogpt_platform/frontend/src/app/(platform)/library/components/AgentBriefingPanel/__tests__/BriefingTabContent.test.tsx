import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BriefingTabContent } from "../BriefingTabContent";

const mockUseGetV2GetCopilotUsage = vi.fn();
const mockUseGetV1UserCostSummary = vi.fn();
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
vi.mock(
  "@/app/api/__generated__/endpoints/graphs/graphs",
  async (importOriginal) => {
    const actual =
      await importOriginal<
        typeof import("@/app/api/__generated__/endpoints/graphs/graphs")
      >();
    return {
      ...actual,
      useGetV1UserCostSummary: (
        _params: unknown,
        opts: { query?: { select?: (r: { data: unknown }) => unknown } },
      ) => {
        const ret = mockUseGetV1UserCostSummary() as { data?: unknown };
        if (
          ret?.data !== undefined &&
          typeof opts?.query?.select === "function"
        ) {
          return { ...ret, data: opts.query.select({ data: ret.data }) };
        }
        return ret;
      },
    };
  },
);

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
  mockUseGetV1UserCostSummary.mockReset();
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

function emptyCostSummary() {
  return {
    data: {
      total_cents: 0,
      run_count: 0,
      billable_run_count: 0,
      failed_cost_cents: 0,
      by_agent: [],
      top_runs: [],
      daily: [],
    },
    isLoading: false,
    isError: false,
  };
}

function makeAgent(overrides: Partial<LibraryAgent> = {}): LibraryAgent {
  return {
    id: overrides.id ?? "lib-1",
    graph_id: overrides.graph_id ?? "g-1",
    name: overrides.name ?? "Agent One",
    image_url: overrides.image_url ?? null,
    description: "",
    creator_image_url: "",
    creator_name: "",
    has_external_trigger: false,
    is_scheduled: false,
    next_scheduled_run: null,
    recommended_schedule_cron: null,
    status: "COMPLETED",
    updated_at: new Date().toISOString(),
    new_output: false,
    can_access_graph: true,
    is_latest_version: true,
    graph_version: 1,
  } as unknown as LibraryAgent;
}

describe("BriefingTabContent — UsageSection", () => {
  it("renders no usage block when usage fetch has not succeeded", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
    });
    mockUseGetV1UserCostSummary.mockReturnValue(emptyCostSummary());
    mockUseGetFlag.mockReturnValue(false);
    render(<BriefingTabContent activeTab="all" agents={[]} />);
    expect(screen.queryByText("Usage limits")).toBeNull();
  });

  it("renders no usage block when both windows are null", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: null, weeklyPercent: null }),
      isSuccess: true,
    });
    mockUseGetV1UserCostSummary.mockReturnValue(emptyCostSummary());
    mockUseGetFlag.mockReturnValue(false);
    render(<BriefingTabContent activeTab="all" agents={[]} />);
    expect(screen.queryByText("Usage limits")).toBeNull();
  });

  it("renders tier badge + daily+weekly meters at normal usage", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 12, weeklyPercent: 4, tier: "PRO" }),
      isSuccess: true,
    });
    mockUseGetV1UserCostSummary.mockReturnValue(emptyCostSummary());
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

  it("shows 'Manage billing' when billing flag is on", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 100, weeklyPercent: 40 }),
      isSuccess: true,
    });
    mockUseGetV1UserCostSummary.mockReturnValue(emptyCostSummary());
    mockUseGetFlag.mockReturnValue(true);
    render(<BriefingTabContent activeTab="all" agents={[]} />);
    expect(screen.getByText("Manage billing")).toBeDefined();
  });

  it("hides 'Manage billing' when billing flag is off", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 100, weeklyPercent: 40 }),
      isSuccess: true,
    });
    mockUseGetV1UserCostSummary.mockReturnValue(emptyCostSummary());
    mockUseGetFlag.mockReturnValue(false);
    render(<BriefingTabContent activeTab="all" agents={[]} />);
    expect(screen.queryByText("Manage billing")).toBeNull();
  });

  it("renders <1% used when percent is >0 but rounds to 0", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: makeUsage({ dailyPercent: 0.4, weeklyPercent: 0 }),
      isSuccess: true,
    });
    mockUseGetV1UserCostSummary.mockReturnValue(emptyCostSummary());
    mockUseGetFlag.mockReturnValue(false);
    render(<BriefingTabContent activeTab="all" agents={[]} />);
    expect(screen.getByText("<1% used")).toBeDefined();
  });

  it("dispatches to ExecutionListSection for running/attention/completed tabs", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
    });
    mockUseGetV1UserCostSummary.mockReturnValue(emptyCostSummary());
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
    mockUseGetV1UserCostSummary.mockReturnValue(emptyCostSummary());
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

describe("BriefingTabContent — CostsBreakdown", () => {
  it("shows 'No spend this month yet' when total is zero", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
    });
    mockUseGetV1UserCostSummary.mockReturnValue(emptyCostSummary());
    mockUseGetFlag.mockReturnValue(false);
    render(<BriefingTabContent activeTab="all" agents={[]} />);
    fireEvent.click(
      screen.getByRole("button", { name: /see costs breakdown/i }),
    );
    expect(screen.getByText("No spend this month yet.")).toBeDefined();
  });

  it("renders headline stats, top runs, and by-agent sections when spend > 0", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
    });
    mockUseGetV1UserCostSummary.mockReturnValue({
      data: {
        total_cents: 4250,
        run_count: 10,
        billable_run_count: 10,
        failed_cost_cents: 50,
        by_agent: [
          { graph_id: "g-1", cost_cents: 3000, run_count: 6 },
          { graph_id: "g-2", cost_cents: 1250, run_count: 4 },
        ],
        top_runs: [
          {
            execution_id: "exec-big",
            graph_id: "g-1",
            cost_cents: 2500,
            started_at: new Date().toISOString(),
            status: "COMPLETED",
            duration_seconds: 30,
            node_error_count: 0,
          },
          {
            execution_id: "exec-small",
            graph_id: "g-2",
            cost_cents: 250,
            started_at: new Date().toISOString(),
            status: "FAILED",
            duration_seconds: 5,
            node_error_count: 2,
          },
        ],
        daily: [
          { date: "2026-05-10", cost_cents: 3000, run_count: 6 },
          { date: "2026-05-11", cost_cents: 1250, run_count: 4 },
        ],
      },
      isLoading: false,
      isError: false,
    });
    mockUseGetFlag.mockReturnValue(false);

    render(
      <BriefingTabContent
        activeTab="all"
        agents={[
          makeAgent({ id: "lib-1", graph_id: "g-1", name: "Alpha" }),
          makeAgent({ id: "lib-2", graph_id: "g-2", name: "Beta" }),
        ]}
      />,
    );

    // Sections are hidden behind the toggle by default
    expect(screen.queryByText("$42.50")).toBeNull();
    fireEvent.click(
      screen.getByRole("button", { name: /see costs breakdown/i }),
    );

    // Headline stats
    expect(screen.getByText("$42.50")).toBeDefined();
    expect(screen.getByText("Most expensive runs")).toBeDefined();
    expect(screen.getByText("Spend by agent")).toBeDefined();

    // Agent names resolved via graph_id lookup
    expect(screen.getAllByText("Alpha").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Beta").length).toBeGreaterThanOrEqual(1);

    // Failure indicator on the FAILED run
    expect(screen.getByText(/2 errors/)).toBeDefined();
  });

  it("surfaces an inline error when the cost-summary endpoint fails", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
    });
    mockUseGetV1UserCostSummary.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
    });
    mockUseGetFlag.mockReturnValue(false);

    render(<BriefingTabContent activeTab="all" agents={[]} />);
    fireEvent.click(
      screen.getByRole("button", { name: /see costs breakdown/i }),
    );
    expect(screen.getByText(/Couldn't load cost breakdown/i)).toBeDefined();
  });

  it("falls back to short graph_id label when agent isn't in the library", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
    });
    mockUseGetV1UserCostSummary.mockReturnValue({
      data: {
        total_cents: 500,
        run_count: 1,
        billable_run_count: 1,
        failed_cost_cents: 0,
        by_agent: [
          { graph_id: "deadbeef-1234-5678", cost_cents: 500, run_count: 1 },
        ],
        top_runs: [],
        daily: [],
      },
      isLoading: false,
      isError: false,
    });
    mockUseGetFlag.mockReturnValue(false);

    render(<BriefingTabContent activeTab="all" agents={[]} />);
    fireEvent.click(
      screen.getByRole("button", { name: /see costs breakdown/i }),
    );
    expect(screen.getByText(/Agent deadbeef/)).toBeDefined();
  });

  it("computes Avg / run from billable_run_count, not run_count", () => {
    mockUseGetV2GetCopilotUsage.mockReturnValue({
      data: undefined,
      isSuccess: false,
    });
    // 6 total runs but only 2 with cost > 0 — avg must be $5.00 (1000/2),
    // not $1.67 (1000/6) which is what dividing by run_count would yield.
    mockUseGetV1UserCostSummary.mockReturnValue({
      data: {
        total_cents: 1000,
        run_count: 6,
        billable_run_count: 2,
        failed_cost_cents: 0,
        by_agent: [{ graph_id: "g-1", cost_cents: 1000, run_count: 6 }],
        top_runs: [],
        daily: [],
      },
      isLoading: false,
      isError: false,
    });
    mockUseGetFlag.mockReturnValue(false);

    render(
      <BriefingTabContent
        activeTab="all"
        agents={[makeAgent({ id: "lib-1", graph_id: "g-1", name: "Alpha" })]}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: /see costs breakdown/i }),
    );

    expect(screen.getByText("$5.00")).toBeDefined();
    expect(screen.queryByText("$1.67")).toBeNull();
  });
});
