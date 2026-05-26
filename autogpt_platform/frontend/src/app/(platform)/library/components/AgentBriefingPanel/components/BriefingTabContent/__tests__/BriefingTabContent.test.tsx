import { getListCopilotSkillsMockHandler } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import {
  getGetV1ListExecutionSchedulesForAUserMockHandler,
  getListCopilotFollowupSchedulesMockHandler,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { CopilotSkillInfo } from "@/app/api/__generated__/models/copilotSkillInfo";
import type { CopilotTurnJobInfo } from "@/app/api/__generated__/models/copilotTurnJobInfo";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BriefingTabContent } from "../BriefingTabContent";

const mockUseGetV1UserCostSummary = vi.fn();
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

afterEach(() => {
  mockUseGetV1UserCostSummary.mockReset();
  server.resetHandlers();
});

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

describe("BriefingTabContent — dispatching", () => {
  it("renders the costs breakdown toggle on the 'all' tab and no AutoPilot usage limits", () => {
    mockUseGetV1UserCostSummary.mockReturnValue(emptyCostSummary());
    render(<BriefingTabContent activeTab="all" agents={[]} />);
    expect(
      screen.getByRole("button", { name: /see costs breakdown/i }),
    ).toBeDefined();
    // AutoPilot rate-limit meters live in the Copilot section, not here.
    expect(screen.queryByText("Usage limits")).toBeNull();
    expect(screen.queryByText("Today")).toBeNull();
    expect(screen.queryByText("This week")).toBeNull();
    expect(screen.queryByText("Manage billing")).toBeNull();
  });

  it("dispatches to ExecutionListSection for running/attention/completed tabs", () => {
    mockUseGetV1UserCostSummary.mockReturnValue(emptyCostSummary());

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
    mockUseGetV1UserCostSummary.mockReturnValue(emptyCostSummary());

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
    mockUseGetV1UserCostSummary.mockReturnValue(emptyCostSummary());
    render(<BriefingTabContent activeTab="all" agents={[]} />);
    fireEvent.click(
      screen.getByRole("button", { name: /see costs breakdown/i }),
    );
    expect(screen.getByText("No spend this month yet.")).toBeDefined();
  });

  it("renders headline stats, top runs, and by-agent sections when spend > 0", () => {
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
    expect(screen.getByText("Most expensive tasks")).toBeDefined();
    expect(screen.getByText("Spend by agent")).toBeDefined();

    // Agent names resolved via graph_id lookup
    expect(screen.getAllByText("Alpha").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("Beta").length).toBeGreaterThanOrEqual(1);

    // Failure indicator on the FAILED run
    expect(screen.getByText(/2 errors/)).toBeDefined();
  });

  it("surfaces an inline error when the cost-summary endpoint fails", () => {
    mockUseGetV1UserCostSummary.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
    });

    render(<BriefingTabContent activeTab="all" agents={[]} />);
    fireEvent.click(
      screen.getByRole("button", { name: /see costs breakdown/i }),
    );
    expect(screen.getByText(/Couldn't load cost breakdown/i)).toBeDefined();
  });

  it("falls back to short graph_id label when agent isn't in the library", () => {
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

    render(<BriefingTabContent activeTab="all" agents={[]} />);
    fireEvent.click(
      screen.getByRole("button", { name: /see costs breakdown/i }),
    );
    expect(screen.getByText(/Agent deadbeef/)).toBeDefined();
  });

  it("computes Avg / run from billable_run_count, not run_count", () => {
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

  it("scales spend-by-agent bars as share of total monthly spend and hides zero-cost rollups", () => {
    // Alpha: 75% of total, Beta: 25% of total, Gamma: $0.00 — should be hidden.
    mockUseGetV1UserCostSummary.mockReturnValue({
      data: {
        total_cents: 4000,
        run_count: 12,
        billable_run_count: 10,
        failed_cost_cents: 0,
        by_agent: [
          { graph_id: "g-1", cost_cents: 3000, run_count: 6 },
          { graph_id: "g-2", cost_cents: 1000, run_count: 4 },
          { graph_id: "g-3", cost_cents: 0, run_count: 2 },
        ],
        top_runs: [],
        daily: [],
      },
      isLoading: false,
      isError: false,
    });

    render(
      <BriefingTabContent
        activeTab="all"
        agents={[
          makeAgent({ id: "lib-1", graph_id: "g-1", name: "Alpha" }),
          makeAgent({ id: "lib-2", graph_id: "g-2", name: "Beta" }),
          makeAgent({ id: "lib-3", graph_id: "g-3", name: "Gamma" }),
        ]}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: /see costs breakdown/i }),
    );

    expect(screen.getByText("% of monthly spend")).toBeDefined();
    expect(screen.getByText("75%")).toBeDefined();
    expect(screen.getByText("25%")).toBeDefined();
    // Zero-cost agent omitted entirely.
    expect(screen.queryByText("Gamma")).toBeNull();
  });

  it("renders an avg-per-run pill next to each agent's run count", () => {
    mockUseGetV1UserCostSummary.mockReturnValue({
      data: {
        total_cents: 2000,
        run_count: 4,
        billable_run_count: 4,
        failed_cost_cents: 0,
        by_agent: [{ graph_id: "g-1", cost_cents: 2000, run_count: 4 }],
        top_runs: [],
        daily: [],
      },
      isLoading: false,
      isError: false,
    });

    render(
      <BriefingTabContent
        activeTab="all"
        agents={[makeAgent({ id: "lib-1", graph_id: "g-1", name: "Alpha" })]}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: /see costs breakdown/i }),
    );

    // 2000c / 4 runs = $5.00 per run, shown inline with the run count.
    expect(screen.getByText(/4 runs · avg \$5\.00/)).toBeDefined();
  });

  it("renders the calendar-month range subtitle when expanded", () => {
    mockUseGetV1UserCostSummary.mockReturnValue({
      data: {
        total_cents: 100,
        run_count: 1,
        billable_run_count: 1,
        failed_cost_cents: 0,
        by_agent: [{ graph_id: "g-1", cost_cents: 100, run_count: 1 }],
        top_runs: [],
        daily: [],
      },
      isLoading: false,
      isError: false,
    });

    render(<BriefingTabContent activeTab="all" agents={[]} />);
    fireEvent.click(
      screen.getByRole("button", { name: /see costs breakdown/i }),
    );
    expect(screen.getByText(/Calendar month so far/)).toBeDefined();
    expect(screen.getByText(/today \(UTC\)/)).toBeDefined();
  });
});

describe("BriefingTabContent — CopilotLibrarySummary (Autopilot pill)", () => {
  function setupBriefingMocks() {
    mockUseGetV1UserCostSummary.mockReturnValue(emptyCostSummary());
  }

  function makeSkill(overrides: { name?: string } = {}): CopilotSkillInfo {
    return {
      name: overrides.name ?? "skill-1",
      description: "test skill",
      triggers: [],
      version: "1.0.0",
      updated_at: new Date().toISOString(),
    } as unknown as CopilotSkillInfo;
  }

  function makeFollowup(overrides: { id?: string } = {}): CopilotTurnJobInfo {
    const runAt = new Date(Date.now() + 60 * 60 * 1000);
    return {
      id: overrides.id ?? "f-1",
      name: "copilot-followup",
      user_id: "user-1",
      session_id: "session-abcdef0123",
      message: "ping",
      cron: null,
      run_at: runAt,
      next_run_time: runAt.toISOString(),
      kind: "copilot_turn" as const,
      timezone: "UTC",
      cap_retry_count: 0,
    };
  }

  function makeGraphSchedule(
    overrides: { id?: string } = {},
  ): GraphExecutionJobInfo {
    return {
      id: overrides.id ?? "g-1",
      name: "daily",
      user_id: "user-1",
      graph_id: "graph-abc",
      graph_version: 1,
      agent_name: "Daily agent",
      cron: "0 9 * * *",
      input_data: {},
      next_run_time: new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString(),
      kind: "graph" as const,
      timezone: "UTC",
    };
  }

  it("renders nothing when there are zero skills AND zero scheduled items", async () => {
    setupBriefingMocks();
    server.use(
      getListCopilotSkillsMockHandler([]),
      getListCopilotFollowupSchedulesMockHandler([]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    );

    render(<BriefingTabContent activeTab="all" agents={[]} />);

    // The pill suppresses entirely when both counts are zero — surfacing
    // "0 skills · 0 scheduled" would be noise, not a discovery affordance.
    await waitFor(() => {
      expect(screen.queryByTestId("copilot-library-summary")).toBeNull();
      expect(screen.queryByText("Autopilot library")).toBeNull();
    });
  });

  it("shows only the skills link when scheduled count is zero", async () => {
    setupBriefingMocks();
    server.use(
      getListCopilotSkillsMockHandler([
        makeSkill({ name: "alpha" }),
        makeSkill({ name: "beta" }),
      ]),
      getListCopilotFollowupSchedulesMockHandler([]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    );

    render(<BriefingTabContent activeTab="all" agents={[]} />);

    expect(await screen.findByText("Autopilot library")).toBeDefined();
    expect(screen.getByTestId("copilot-library-skills-link").textContent).toBe(
      "2 skills",
    );
    expect(screen.queryByTestId("copilot-library-followups-link")).toBeNull();
  });

  it("counts copilot follow-ups only (graph schedules are NOT folded in — they have their own briefing tab)", async () => {
    setupBriefingMocks();
    server.use(
      getListCopilotSkillsMockHandler([]),
      getListCopilotFollowupSchedulesMockHandler([
        makeFollowup({ id: "f1" }),
        makeFollowup({ id: "f2" }),
      ]),
      // Graph schedules exist server-side but MUST NOT be added to the
      // pill count — the briefing's own "Scheduled" tab already covers
      // them, so folding them in here would double-count.
      getGetV1ListExecutionSchedulesForAUserMockHandler([
        makeGraphSchedule({ id: "g1" }),
        makeGraphSchedule({ id: "g2" }),
      ]),
    );

    render(<BriefingTabContent activeTab="all" agents={[]} />);

    expect(await screen.findByText("Autopilot library")).toBeDefined();
    expect(
      screen.getByTestId("copilot-library-followups-link").textContent,
    ).toBe("2 follow-ups");
    expect(screen.queryByTestId("copilot-library-skills-link")).toBeNull();
  });

  it("hides the pill entirely when skills=0 AND copilot follow-ups=0 (even if graph schedules exist)", async () => {
    setupBriefingMocks();
    server.use(
      getListCopilotSkillsMockHandler([]),
      getListCopilotFollowupSchedulesMockHandler([]),
      // Graph schedules alone don't count toward the autopilot pill —
      // they belong to the "Scheduled" briefing tab. Pill must stay
      // hidden so we don't surface a bare "Autopilot library" header
      // with nothing actionable.
      getGetV1ListExecutionSchedulesForAUserMockHandler([
        makeGraphSchedule({ id: "g1" }),
      ]),
    );

    render(<BriefingTabContent activeTab="all" agents={[]} />);

    await waitFor(() => {
      expect(screen.queryByTestId("copilot-library-summary")).toBeNull();
      expect(screen.queryByText("Autopilot library")).toBeNull();
    });
  });

  it("shows both links with a separator when skills AND follow-ups are positive (singular pluralization)", async () => {
    setupBriefingMocks();
    server.use(
      getListCopilotSkillsMockHandler([makeSkill({ name: "only-one" })]),
      getListCopilotFollowupSchedulesMockHandler([makeFollowup({ id: "f1" })]),
      getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    );

    render(<BriefingTabContent activeTab="all" agents={[]} />);

    expect(await screen.findByText("Autopilot library")).toBeDefined();
    // Singular form when count is 1 — verifies the pluralization branch.
    expect(screen.getByTestId("copilot-library-skills-link").textContent).toBe(
      "1 skill",
    );
    expect(
      screen.getByTestId("copilot-library-followups-link").textContent,
    ).toBe("1 follow-up");
    // Separator dot is only rendered when BOTH links are visible.
    const pill = screen.getByTestId("copilot-library-summary");
    expect(pill.textContent).toContain("•");
  });
});
