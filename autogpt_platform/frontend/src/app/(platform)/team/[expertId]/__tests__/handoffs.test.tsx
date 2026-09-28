import {
  getGetExpertActivityMockHandler,
  getGetExpertMockHandler,
  getListDelegationsMockHandler200,
  getListExpertRunsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetHomeDashboardMockHandler,
  getGetHomeDashboardResponseMock200,
} from "@/app/api/__generated__/endpoints/home/home.msw";
import { getGetV1ListExecutionSchedulesForAUserMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { DelegationSummary } from "@/app/api/__generated__/models/delegationSummary";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import ExpertDetailPage from "../page";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: string) =>
      flag === "hire-experts"
        ? { enabled: true, ready: true }
        : actual.useFlagStatus(flag as never),
  };
});

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

function withHandoffs(delegations: DelegationSummary[]) {
  server.use(
    getListDelegationsMockHandler200({
      delegations,
      summary: {
        working: 0,
        needs_you: 0,
        completed: 0,
        failed: 0,
        spent_today_usd: 0,
      },
    }),
  );
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

async function openWork() {
  await userEvent.click(await screen.findByRole("tab", { name: "Work" }));
}

describe("an expert's hand-offs from Otto", () => {
  test("lists them above the runs with the week's summary", async () => {
    withHandoffs([
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
    ]);
    render(<ExpertDetailPage />);
    await openWork();

    expect((await screen.findByTestId("handoffs-summary")).textContent).toBe(
      "2 delegations this week · 1 done · 1 failed · $1.67 spent",
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
    expect(screen.getByText("Alex's Work")).toBeDefined();
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
    await openWork();
    const filters = await screen.findByRole("group", {
      name: "Filter hand-offs",
    });

    fireEvent.click(
      within(filters).getByRole("button", { name: "In progress" }),
    );
    expect(screen.getByText("Launch checklist")).toBeDefined();
    expect(screen.queryByText("Onboarding revamp PRD")).toBeNull();

    fireEvent.click(within(filters).getByRole("button", { name: "Failed" }));
    expect(screen.getByText("Nothing here right now.")).toBeDefined();
  });

  test("says Working for Otto in the header while a hand-off runs", async () => {
    withHandoffs([handoff({ status: "running" })]);
    render(<ExpertDetailPage />);

    expect(await screen.findByText("Working for Otto")).toBeDefined();
  });

  test("shows an empty state when Otto has handed nothing over", async () => {
    withHandoffs([]);
    render(<ExpertDetailPage />);
    await openWork();

    expect(await screen.findByText(/Nothing handed to Alex yet/)).toBeDefined();
    expect(screen.queryByText("Working for Otto")).toBeNull();
  });
});
