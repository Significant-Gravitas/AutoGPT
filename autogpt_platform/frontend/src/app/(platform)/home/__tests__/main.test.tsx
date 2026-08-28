import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, expect, test, vi } from "vitest";
import type { HomeAgentStatus } from "@/app/api/__generated__/models/homeAgentStatus";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import HomePage from "../page";

const { setFlagStatusMock } = vi.hoisted(() => ({
  setFlagStatusMock: vi.fn(() => ({ enabled: true, ready: true })),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: string) =>
      flag === "hire-experts"
        ? setFlagStatusMock()
        : actual.useFlagStatus(flag as never),
    useGetFlag: (flag: string) => flag === actual.Flag.HIRE_EXPERTS,
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { user_metadata: { preferred_name: "Abhi" } },
  }),
}));

const notFoundMock = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => "/home",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    notFoundMock();
    throw new Error("NEXT_NOT_FOUND");
  },
}));

const NOW = new Date("2026-08-09T12:00:00Z");
const maria = {
  id: "maria",
  name: "Maria",
  role: "Planning",
  avatar_url: null,
};

const approvalItem: HomeAttentionItem = {
  id: "attention-approval",
  kind: "approval",
  priority: "high",
  title: "Approve the camera shortlist",
  description: "Maria wants to send the shortlist to your inbox.",
  why_it_matters: "The research run is paused until you decide.",
  expert: maria,
  primary_action: { label: "Review", href: "/library/runs/run-1" },
  review: {
    node_exec_id: "node-exec-1",
    graph_exec_id: "graph-exec-1",
    graph_id: "graph-1",
    graph_version: 1,
    user_id: "user-1",
    payload: {},
    editable: false,
    status: "WAITING",
    created_at: NOW,
  },
};

const dashboard: HomeDashboardResponse = {
  generated_at: NOW,
  timezone: "UTC",
  attention: [
    {
      id: "attention-1",
      kind: "setup",
      priority: "high",
      title: "Connect your calendar for Maria",
      description: "Maria cannot schedule your learning sessions yet.",
      why_it_matters: "Three planned sessions are waiting for this connection.",
      preview: "3 days · 7 stops · $620 estimated total",
      expert: maria,
      primary_action: { label: "Finish setup", href: "/team/maria" },
    },
  ],
  briefing: {
    generated_at: NOW,
    window_started_at: new Date("2026-08-08T12:00:00Z"),
    completed_count: 14,
    failed_count: 1,
    routine_count: 13,
    outcomes: [
      {
        id: "outcome-1",
        status: "completed",
        title: "Your camera research is ready",
        summary: "Compared 18 cameras and shortlisted the best three.",
        expert: maria,
        agent_name: "Product Researcher",
        occurred_at: NOW,
        duration_seconds: 812,
        cost_cents: 42,
        link: "/library",
      },
      {
        id: "outcome-2",
        status: "failed",
        title: "Learning sessions could not be scheduled",
        summary: "Nova needs Calendar access before the plan can continue.",
        expert: maria,
        agent_name: "Weekly Learning Coach",
        occurred_at: NOW,
        duration_seconds: 44,
        cost_cents: 3,
        link: "/team/maria",
      },
    ],
  },
  active_tasks: [
    {
      id: "active-1",
      title: "Checking recurring subscriptions",
      status: "running",
      expert: maria,
      started_at: new Date("2026-08-09T11:52:00Z"),
      link: "/library",
    },
  ],
  work_items: [],
  upcoming_tasks: [
    {
      id: "upcoming-1",
      title: "Spanish practice plan",
      kind: "agent",
      expert: maria,
      next_run_time: new Date("2026-08-09T13:20:00Z"),
    },
  ],
  team: { total: 10, ready: 7, working: 1, needs_attention: 2 },
  agents: [
    {
      expert: maria,
      status: "working",
      detail: "Checking recurring subscriptions",
    },
  ],
  week: {
    run_count: 32,
    completed_count: 27,
    review_count: 3,
    failed_count: 2,
    total_runtime_seconds: 18_120,
    timed_run_count: 30,
    total_cost_cents: 367,
    credits_balance: 15_600,
    daily: [
      {
        date: new Date("2026-08-09T00:00:00Z"),
        completed_count: 6,
        review_count: 1,
        failed_count: 0,
      },
    ],
  },
};

afterEach(() => {
  setFlagStatusMock.mockReturnValue({ enabled: true, ready: true });
});

function mockDashboard(response: HomeDashboardResponse) {
  server.use(
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(response),
    ),
  );
}

test("renders every Home tile from the aggregate API", async () => {
  mockDashboard(dashboard);

  render(<HomePage />);

  expect(await screen.findByText(/Abhi/)).toBeDefined();
  expect(
    screen.getByRole("heading", { name: "Needs your decision" }),
  ).toBeDefined();
  expect(screen.getByLabelText("1 item needs your attention")).toBeDefined();
  expect(screen.getByText("Connect your calendar for Maria")).toBeDefined();
  expect(screen.getByRole("heading", { name: "Delivered" })).toBeDefined();
  expect(screen.getByText("Your camera research is ready")).toBeDefined();
  expect(screen.getByText(/13 routine tasks completed quietly/)).toBeDefined();
  expect(screen.getByRole("heading", { name: "Your agents" })).toBeDefined();
  expect(
    screen.getByRole("link", { name: /View all 10 agents/ }),
  ).toBeDefined();
  expect(
    screen.getByRole("heading", { name: "Working now & next" }),
  ).toBeDefined();
  expect(
    screen.getAllByText("Checking recurring subscriptions").length,
  ).toBeGreaterThan(0);
  expect(screen.getByText("Spanish practice plan")).toBeDefined();
});

test("does not expose weekly spend or account balances", async () => {
  mockDashboard({
    ...dashboard,
    team: { ...dashboard.team, spend_cents: 1_250 },
    agents: [
      { ...dashboard.agents[0], spend_cents: 900 },
      {
        expert: { id: "nova", name: "Nova", role: "Ops", avatar_url: null },
        status: "ready",
        detail: "Ready for the next task",
        spend_cents: 0,
      },
    ],
  });

  render(<HomePage />);

  expect(await screen.findByText("Ready for the next task")).toBeDefined();
  expect(screen.queryByText("· $9.00 this week")).toBeNull();
  expect(screen.queryByText(/\$12\.50 this week/)).toBeNull();
  expect(screen.queryByText("· $0.00 this week")).toBeNull();
  expect(screen.queryByText("15,600")).toBeNull();
});

test("redacts internal paths and identifiers from Home work summaries", async () => {
  mockDashboard({
    ...dashboard,
    briefing: {
      ...dashboard.briefing,
      outcomes: [
        {
          ...dashboard.briefing.outcomes[0],
          title: "Report from /tmp/copilot-session/raw.json",
          summary:
            "execution_id=exec-secret for 357dd4a1-acc7-4942-a317-c6d47e5ade6a",
        },
      ],
    },
    active_tasks: [
      {
        ...dashboard.active_tasks[0],
        title: "Building /tmp/copilot-session/build_state.json",
      },
    ],
  });

  render(<HomePage />);

  expect(await screen.findAllByText(/workspace file/)).not.toHaveLength(0);
  expect(screen.queryByText(/\/tmp\/copilot-session/)).toBeNull();
  expect(screen.queryByText(/exec-secret/)).toBeNull();
  expect(screen.queryByText(/357dd4a1-acc7/)).toBeNull();
});

test("falls back to an Unknown badge for an unrecognised agent status", async () => {
  mockDashboard({
    ...dashboard,
    agents: [
      {
        ...dashboard.agents[0],
        status: "decommissioned" as HomeAgentStatus["status"],
      },
    ],
  });

  render(<HomePage />);

  expect(await screen.findByText("Unknown")).toBeDefined();
});

test("approving an item one-tap sends the review decision", async () => {
  const user = userEvent.setup();
  const reviewRequests: unknown[] = [];
  mockDashboard({ ...dashboard, attention: [approvalItem] });
  server.use(
    http.post("/api/proxy/api/review/action", async ({ request }) => {
      reviewRequests.push(await request.json());
      return HttpResponse.json({ failed_count: 0, processed_count: 1 });
    }),
  );

  render(<HomePage />);

  await user.click(
    await screen.findByRole("button", {
      name: "Approve: Approve the camera shortlist",
    }),
  );

  await waitFor(() => expect(reviewRequests).toHaveLength(1));
  expect(reviewRequests[0]).toEqual({
    reviews: [
      {
        node_exec_id: "node-exec-1",
        approved: true,
        auto_approve_future: false,
      },
    ],
  });
});

test("keeps a Review deep link alongside the approval shortcuts", async () => {
  mockDashboard({ ...dashboard, attention: [approvalItem] });

  render(<HomePage />);

  const reviewLink = await screen.findByRole("link", { name: "Review" });
  expect(reviewLink.getAttribute("href")).toBe("/library/runs/run-1");
});

test("guides a clean-slate user to build their first AI team", async () => {
  mockDashboard({
    ...dashboard,
    attention: [],
    briefing: {
      ...dashboard.briefing,
      completed_count: 0,
      failed_count: 0,
      routine_count: 0,
      outcomes: [],
    },
    active_tasks: [],
    upcoming_tasks: [],
    team: { total: 0, ready: 0, working: 0, needs_attention: 0 },
    agents: [],
  });

  render(<HomePage />);

  expect(
    await screen.findByRole("heading", { name: "Build your first AI team" }),
  ).toBeDefined();
  expect(screen.getByText("Product")).toBeDefined();
  expect(screen.getByText("Engineering")).toBeDefined();
  expect(screen.getByText("Marketing")).toBeDefined();
  expect(screen.getByText("Sales")).toBeDefined();
  expect(
    screen
      .getByRole("link", { name: "Start with AutoPilot" })
      .getAttribute("href"),
  ).toBe("/copilot");
  expect(screen.queryByText("You are all caught up")).toBeNull();
});

test("keeps failed work in Risks instead of Delivered", async () => {
  mockDashboard(dashboard);

  render(<HomePage />);

  expect(
    await screen.findByText("Learning sessions could not be scheduled"),
  ).toBeDefined();
  expect(screen.getByText("Your camera research is ready")).toBeDefined();
  expect(
    screen.queryByRole("button", { name: "Filter briefing outcomes: All" }),
  ).toBeNull();
});

test("keeps partial workflow delivery in Risks instead of Delivered", async () => {
  mockDashboard({
    ...dashboard,
    briefing: {
      ...dashboard.briefing,
      outcomes: [
        {
          ...dashboard.briefing.outcomes[1],
          status: "partial",
          title: "Lead research needs attention",
          summary: "One required workflow step failed.",
        },
      ],
    },
  });

  render(<HomePage />);

  expect(
    await screen.findByText("Lead research needs attention"),
  ).toBeDefined();
  expect(screen.getByText("Nothing delivered yet")).toBeDefined();
});

test("shows delegated risk confidence and scrubs internal details", async () => {
  mockDashboard({
    ...dashboard,
    work_items: [
      {
        id: "work-1",
        title: "Review /tmp/internal/search.json",
        expected_deliverable: "A safe recommendation",
        status: "blocked_manager",
        expert: maria,
        progress: 60,
        blocker: 'tool_call_id=abc graph_id=def {"query":"private payload"}',
        confidence: "likely",
        artifacts: [],
        updated_at: NOW,
        link: "/team/maria?workItemId=work-1#work-item-work-1",
      },
    ],
  });

  render(<HomePage />);

  const riskTitle = await screen.findByText("Review a workspace file");
  const risk = riskTitle.closest("a");
  expect(risk).not.toBeNull();
  expect(risk?.getAttribute("href")).toBe(
    "/team/maria?workItemId=work-1#work-item-work-1",
  );
  expect(screen.queryByText(/private payload/)).toBeNull();
  expect(screen.queryByText(/tool_call_id/)).toBeNull();
});

test("shows a retryable page error when the aggregate cannot load", async () => {
  let attempts = 0;
  server.use(
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () => {
      attempts += 1;
      return attempts === 1
        ? HttpResponse.json({ detail: "boom" }, { status: 500 })
        : HttpResponse.json(dashboard);
    }),
  );

  const user = userEvent.setup();
  render(<HomePage />);

  expect(
    await screen.findByText("Your Home briefing could not be loaded"),
  ).toBeDefined();

  await user.click(screen.getByRole("button", { name: /try again/i }));

  expect(
    await screen.findByRole("heading", { name: "Needs your decision" }),
  ).toBeDefined();
});

test("calls notFound when the experts feature is disabled", () => {
  mockDashboard(dashboard);
  setFlagStatusMock.mockReturnValueOnce({ enabled: false, ready: true });
  notFoundMock.mockClear();

  try {
    render(<HomePage />);
  } catch {}
  expect(notFoundMock).toHaveBeenCalled();
});

test("opens the briefing with the AI-written narrative when there is one", async () => {
  mockDashboard({
    ...dashboard,
    briefing: {
      ...dashboard.briefing,
      narrative:
        "I finished your camera research overnight and one scheduling run needs a retry.",
    },
  });

  render(<HomePage />);

  expect(
    await screen.findByText(
      "I finished your camera research overnight and one scheduling run needs a retry.",
    ),
  ).toBeDefined();
  expect(screen.getByText("Your camera research is ready")).toBeDefined();
});

test("renders the briefing unchanged when no narrative was generated", async () => {
  mockDashboard({ ...dashboard, briefing: { ...dashboard.briefing } });

  render(<HomePage />);

  expect(
    await screen.findByRole("heading", { name: "Delivered" }),
  ).toBeDefined();
  expect(screen.getByText("Your camera research is ready")).toBeDefined();
});
