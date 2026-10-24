import {
  getGetExpertActivityMockHandler,
  getGetExpertMockHandler,
  getListExpertRunsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetHomeDashboardMockHandler,
  getGetHomeDashboardResponseMock200,
} from "@/app/api/__generated__/endpoints/home/home.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { DelegationSummary } from "@/app/api/__generated__/models/delegationSummary";
import { listDelegationsLikeTheServer } from "../../autopilot/__tests__/delegationFixtures";
import {
  type DelegationGroup,
  getDelegationGroup,
} from "../../components/DelegationList/helpers";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";
import ExpertDetailPage from "../page";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/team/expert-alex",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ expertId: "expert-alex" }),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

const alex = {
  id: "expert-alex",
  name: "Alex",
  avatar_url: null,
  role: "Product Manager",
  bio: null,
  skills: [],
  tagline: null,
  identity: "You are Alex.",
  voice_preferences: null,
  boundaries: null,
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
} as unknown as Expert;

const NOW = new Date();
const LAST_WEEK = new Date(NOW.getTime() - 3 * 24 * 60 * 60 * 1000);

function handoff(over: Partial<DelegationSummary>): DelegationSummary {
  return {
    sub_session_id: "sub-1",
    parent_session_id: "otto-chat",
    expert: { id: "expert-alex", name: "Alex", role: "Product Manager" },
    title: "Onboarding revamp PRD",
    brief: "PRD draft with 9 acceptance criteria and 3 open questions.",
    status: "completed",
    created_at: NOW,
    finished_at: NOW,
    elapsed_seconds: 400,
    cost_usd: 0.31,
    files_count: 1,
    question: null,
    question_options: [],
    delegated_by_expert_id: null,
    ...over,
  };
}

function withHandoffs(delegations: DelegationSummary[], spentToday = 0) {
  const listed = vi.fn();
  server.use(
    listDelegationsLikeTheServer(
      delegations,
      {
        working: count(delegations, "working"),
        needs_you: count(delegations, "needs_you"),
        completed: count(delegations, "completed"),
        failed: count(delegations, "failed"),
        started_today: 0,
        spent_today_usd: spentToday,
      },
      listed,
    ),
  );
  return listed;
}

function count(rows: DelegationSummary[], group: DelegationGroup) {
  return rows.filter((row) => getDelegationGroup(row.status) === group).length;
}

beforeEach(() => {
  server.use(
    getGetHomeDashboardMockHandler(
      getGetHomeDashboardResponseMock200({ attention: [] }),
    ),
    getGetExpertMockHandler(alex),
    getGetV1ListExecutionSchedulesForAUserMockHandler([]),
    getListExpertRunsMockHandler([]),
    getGetExpertActivityMockHandler({ timezone: "UTC", days: [] }),
  );
});

describe("an expert's hand-offs from Otto", () => {
  test("lists them on the Overview with the server's summary", async () => {
    withHandoffs(
      [
        handoff({}),
        handoff({
          sub_session_id: "sub-2",
          title: "Roadmap re-prioritisation",
          brief: "Stopped before the scoring step.",
          status: "failed",
          created_at: LAST_WEEK,
          elapsed_seconds: 250,
          cost_usd: 1.36,
          files_count: 0,
        }),
      ],
      1.67,
    );
    render(<ExpertDetailPage />);

    expect((await screen.findByTestId("handoffs-summary")).textContent).toBe(
      "2 delegations · 1 done · 1 failed · $1.67 spent today",
    );
    expect(screen.getByText("All from Otto")).toBeDefined();
    const rows = within(
      screen.getByRole("list", { name: "Hand-offs" }),
    ).getAllByRole("listitem");
    expect(rows).toHaveLength(2);
    expect(
      within(rows[0]).getByText(
        /^From Otto · today \d\d:\d\d · 6m 40s · \$0\.31 · 1 file$/,
      ),
    ).toBeDefined();
    expect(within(rows[0]).getByText("Done")).toBeDefined();
    expect(within(rows[0]).getByRole("link").getAttribute("href")).toBe(
      "/copilot?sessionId=otto-chat",
    );
  });

  test("filters to what is in progress or failed", async () => {
    withHandoffs([
      handoff({}),
      handoff({
        sub_session_id: "sub-2",
        title: "Launch checklist",
        status: "running",
      }),
    ]);
    render(<ExpertDetailPage />);
    const filters = await screen.findByRole("group", {
      name: "Filter hand-offs",
    });

    fireEvent.click(
      within(filters).getByRole("button", { name: "In progress" }),
    );
    await waitFor(() =>
      expect(screen.queryByText("Onboarding revamp PRD")).toBeNull(),
    );
    expect(screen.getByText("Launch checklist")).toBeDefined();

    fireEvent.click(within(filters).getByRole("button", { name: "Failed" }));
    expect(await screen.findByText("Nothing here right now.")).toBeDefined();
  });

  test("says Working for Otto in the header while a hand-off runs", async () => {
    withHandoffs([handoff({ status: "running" })]);
    render(<ExpertDetailPage />);

    expect(await screen.findByText("Working for Otto")).toBeDefined();
  });

  test("leaves the Overview as it was when Otto has handed nothing over", async () => {
    const listed = withHandoffs([]);
    render(<ExpertDetailPage />);

    expect(await screen.findByText("Conversations")).toBeDefined();
    await waitFor(() => expect(listed).toHaveBeenCalled());
    expect(
      screen.queryByRole("region", { name: "Hand-offs from Otto" }),
    ).toBeNull();
    expect(screen.queryByText(/Nothing handed to Alex yet/)).toBeNull();
    expect(screen.queryByText("Working for Otto")).toBeNull();
  });
});
