import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import type { HomeRecentWork } from "@/app/api/__generated__/models/homeRecentWork";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import HomePage from "../page";

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: string) =>
      flag === "hire-experts"
        ? { enabled: true, ready: true }
        : actual.useFlagStatus(flag as never),
    useGetFlag: (flag: string) => flag === actual.Flag.HIRE_EXPERTS,
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: { user_metadata: { preferred_name: "Abhi" } } }),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
  usePathname: () => "/home",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

const NOW = new Date("2026-09-28T10:50:00Z");
const alex = {
  id: "exp-alex",
  name: "Alex",
  role: "Product Manager",
  avatar_url: null,
};
const devon = { ...alex, id: "exp-devon", name: "Devon", role: "Engineer" };

const delegatedQuestion: HomeAttentionItem = {
  id: "question-sub-1",
  kind: "question",
  priority: "normal",
  title: "Q4 release train or December mini-launch?",
  description:
    "Alex, working for Otto on “Onboarding revamp PRD” · asked 3m ago",
  why_it_matters: "The hand-off is paused until you answer.",
  expert: alex,
  created_at: NOW,
  primary_action: { label: "Answer", href: "/copilot?sessionId=otto-chat" },
};

const handoffApproval: HomeAttentionItem = {
  id: "approval-gate-1",
  kind: "approval",
  priority: "normal",
  title: "Hand off “Retention policy check” to Devon",
  headline: { ask: "Hand a task to", object: "Devon" },
  description: "Otto → Devon · ask-first mode",
  why_it_matters: "Nothing runs until you approve it.",
  expert: devon,
  created_at: NOW,
  review: {
    node_exec_id: "gate-1",
    user_id: "user-1",
    session_id: "otto-chat",
    payload: { handoff: { expert_id: "exp-devon", brief: "Retention check" } },
    editable: false,
    status: "WAITING",
    created_at: NOW,
  },
  primary_action: { label: "Open chat", href: "/copilot?sessionId=otto-chat" },
};

const recentWork: HomeRecentWork = {
  window_started_at: new Date("2026-09-21T10:00:00Z"),
  completed_count: 1,
  failed_count: 1,
  total_count: 2,
  groups: [
    {
      actor: { kind: "expert", name: "Alex", expert: alex },
      latest_at: NOW,
      delegation_count: 2,
      delegations: [
        {
          id: "delegation-sub-1",
          sub_session_id: "sub-1",
          title: "Onboarding revamp PRD",
          description: "Delegated 10:41 · returned 10:48 · 1 file",
          status: "completed",
          expert: alex,
          occurred_at: NOW,
          files_count: 1,
          cost_usd: 0.31,
          link: "/copilot?sessionId=otto-chat",
        },
        {
          id: "delegation-sub-2",
          sub_session_id: "sub-2",
          title: "Partner outreach list",
          description: "Delegated 09:10 · failed 09:14",
          status: "failed",
          expert: alex,
          occurred_at: NOW,
        },
      ],
    },
  ],
};

function mockDashboard(
  attention: HomeAttentionItem[],
  recent?: HomeRecentWork,
) {
  const dashboard: HomeDashboardResponse = {
    generated_at: NOW,
    timezone: "UTC",
    attention,
    briefing: {
      generated_at: NOW,
      window_started_at: new Date("2026-09-27T10:00:00Z"),
      completed_count: 0,
      failed_count: 0,
      routine_count: 0,
      outcomes: [],
      author: { kind: "autopilot", name: "Otto", role: "Head of AI" },
    },
    active_tasks: [],
    upcoming_tasks: [],
    team: { total: 0, ready: 0, working: 0, needs_attention: 0 },
    agents: [],
    week: {
      run_count: 0,
      completed_count: 0,
      review_count: 0,
      failed_count: 0,
      total_runtime_seconds: 0,
      timed_run_count: 0,
      total_cost_cents: 0,
      credits_balance: 0,
      daily: [],
    },
    recent_work: recent,
  };
  server.use(
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(dashboard),
    ),
  );
}

test("a teammate's question for Otto links to Otto's chat to answer", async () => {
  mockDashboard([delegatedQuestion]);
  render(<HomePage />);

  const title = await screen.findByText(
    "Q4 release train or December mini-launch?",
  );
  const row = title.closest("article") as HTMLElement;
  expect(within(row).getByText("Question")).toBeDefined();
  expect(
    within(row).getByText(
      "Alex, working for Otto on “Onboarding revamp PRD” · asked 3m ago",
    ),
  ).toBeDefined();
  expect(
    within(row).getByRole("link", { name: "Answer" }).getAttribute("href"),
  ).toBe("/copilot?sessionId=otto-chat");
});

test("a held hand-off keeps its own title and approves in one tap", async () => {
  const user = userEvent.setup();
  const reviewRequests: unknown[] = [];
  mockDashboard([handoffApproval]);
  server.use(
    http.post("/api/proxy/api/review/action", async ({ request }) => {
      reviewRequests.push(await request.json());
      return HttpResponse.json({ failed_count: 0, processed_count: 1 });
    }),
  );
  render(<HomePage />);

  const title = await screen.findByText(
    "Hand off “Retention policy check” to Devon",
  );
  const row = title.closest("article") as HTMLElement;
  expect(within(row).getByText("Approval")).toBeDefined();

  const approve = within(row).getByRole("button", {
    name: "Approve: Hand off “Retention policy check” to Devon",
  });
  expect(approve.textContent).toBe("Approve");
  await user.click(approve);

  await waitFor(() => expect(reviewRequests).toHaveLength(1));
  expect(reviewRequests[0]).toEqual({
    reviews: [
      { node_exec_id: "gate-1", approved: true, auto_approve_future: false },
    ],
  });
});

test("recent work lists the week's hand-offs under the teammate who took them", async () => {
  mockDashboard([], recentWork);
  render(<HomePage />);

  const group = await screen.findByRole("group", { name: "Hand-offs to Alex" });
  const done = within(group).getByText("Onboarding revamp PRD").closest("a");
  expect(done?.getAttribute("href")).toBe("/copilot?sessionId=otto-chat");
  expect(
    within(group).getByText("Delegated 10:41 · returned 10:48 · 1 file"),
  ).toBeDefined();
  expect(within(group).getByText("Done")).toBeDefined();
  expect(within(group).getByText("Failed")).toBeDefined();
  expect(screen.getByText("2 hand-offs")).toBeDefined();
});
