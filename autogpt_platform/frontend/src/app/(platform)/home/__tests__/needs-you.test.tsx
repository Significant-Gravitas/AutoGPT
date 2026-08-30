import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
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

const NOW = new Date("2026-08-09T12:00:00Z");
const maria = {
  id: "maria",
  name: "Maria",
  role: "Planning",
  avatar_url: null,
};

function makeApproval(index: number): HomeAttentionItem {
  return {
    id: `approval-${index}`,
    kind: "approval",
    priority: "high",
    title: `Approve item ${index}`,
    description: "Maria is waiting on your decision.",
    why_it_matters: "The run is paused until you decide.",
    expert: maria,
    primary_action: { label: "Review", href: `/library/runs/run-${index}` },
    review: {
      node_exec_id: `node-exec-${index}`,
      graph_exec_id: `graph-exec-${index}`,
      graph_id: "graph-1",
      graph_version: 1,
      user_id: "user-1",
      payload: {},
      editable: false,
      status: "WAITING",
      created_at: NOW,
    },
  };
}

const setupItem: HomeAttentionItem = {
  id: "setup-1",
  kind: "setup",
  priority: "normal",
  title: "Connect your calendar",
  description: "Maria cannot schedule sessions yet.",
  why_it_matters: "Sessions are waiting on this connection.",
  expert: maria,
  primary_action: { label: "Finish setup", href: "/team/maria" },
};

const questionItem: HomeAttentionItem = {
  id: "question-sess-1",
  kind: "question",
  priority: "normal",
  title: "Maria has a question",
  description: "Monday morning or Friday evening?",
  why_it_matters: "The work is paused until you answer in the chat.",
  expert: maria,
  created_at: NOW,
  primary_action: { label: "Answer", href: "/copilot?sessionId=sess-1" },
};

function makeDashboard(attention: HomeAttentionItem[]): HomeDashboardResponse {
  return {
    generated_at: NOW,
    timezone: "UTC",
    attention,
    briefing: {
      generated_at: NOW,
      window_started_at: new Date("2026-08-08T12:00:00Z"),
      completed_count: 0,
      failed_count: 0,
      routine_count: 0,
      outcomes: [],
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
  };
}

function mockDashboard(attention: HomeAttentionItem[]) {
  server.use(
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(makeDashboard(attention)),
    ),
  );
}

test("lists every attention item without collapsing", async () => {
  mockDashboard([1, 2, 3, 4].map(makeApproval));

  render(<HomePage />);

  expect(await screen.findByText("Approve item 1")).toBeDefined();
  expect(screen.getByText("Approve item 4")).toBeDefined();
});

test("filters the attention list by kind", async () => {
  const user = userEvent.setup();
  mockDashboard([makeApproval(1), setupItem]);

  render(<HomePage />);

  await user.click(
    await screen.findByRole("button", { name: "Filter interventions: All" }),
  );
  await user.click(screen.getByRole("menuitemradio", { name: "Setup" }));

  expect(screen.getByText("Connect your calendar")).toBeDefined();
  expect(screen.queryByText("Approve item 1")).toBeNull();
});

test("requires a second press to confirm a decline", async () => {
  const user = userEvent.setup();
  const reviewRequests: unknown[] = [];
  mockDashboard([makeApproval(1)]);
  server.use(
    http.post("/api/proxy/api/review/action", async ({ request }) => {
      reviewRequests.push(await request.json());
      return HttpResponse.json({ failed_count: 0, processed_count: 1 });
    }),
  );

  render(<HomePage />);

  await user.click(
    await screen.findByRole("button", { name: "Decline: Approve item 1" }),
  );
  expect(reviewRequests).toHaveLength(0);

  await user.click(
    screen.getByRole("button", { name: "Confirm decline: Approve item 1" }),
  );

  await waitFor(() => expect(reviewRequests).toHaveLength(1));
  expect(reviewRequests[0]).toEqual({
    reviews: [
      {
        node_exec_id: "node-exec-1",
        approved: false,
        auto_approve_future: false,
      },
    ],
  });
});

test("keeps a rejected review actionable instead of dropping the row", async () => {
  const user = userEvent.setup();
  mockDashboard([makeApproval(1)]);
  server.use(
    http.post("/api/proxy/api/review/action", () =>
      HttpResponse.json({
        failed_count: 1,
        processed_count: 0,
        error: "Run already finished",
      }),
    ),
  );

  render(<HomePage />);

  const approve = await screen.findByRole("button", {
    name: "Approve: Approve item 1",
  });
  await user.click(approve);

  await waitFor(() => expect(approve.hasAttribute("disabled")).toBe(false));
  expect(screen.getByText("Approve item 1")).toBeDefined();
});

test("shows an unanswered copilot question and links back to the chat", async () => {
  mockDashboard([questionItem]);

  render(<HomePage />);

  expect(await screen.findByText("Maria has a question")).toBeDefined();
  expect(screen.getByText("Monday morning or Friday evening?")).toBeDefined();
  expect(
    screen.getByRole("link", { name: "Answer" }).getAttribute("href"),
  ).toBe("/copilot?sessionId=sess-1");
});

test("offers no approve or decline on a question", async () => {
  mockDashboard([questionItem]);

  render(<HomePage />);

  await screen.findByText("Maria has a question");
  expect(
    screen.queryByRole("button", { name: /Approve: Maria has a question/ }),
  ).toBeNull();
  expect(
    screen.queryByRole("button", { name: /Decline: Maria has a question/ }),
  ).toBeNull();
});

const escalationItem: HomeAttentionItem = {
  id: "task-escalation-task-1",
  kind: "task_escalation",
  priority: "high",
  title: "Maria needs a decision on “Launch email”",
  description: "Ship to staging or prod?",
  why_it_matters: "The task is paused until you answer.",
  expert: maria,
  created_at: NOW,
  task_id: "task-1",
  options: ["Staging", "Prod"],
  primary_action: { label: "View task", href: "/team?task=task-1" },
};

function mockAnswerEndpoint(answerRequests: unknown[]) {
  server.use(
    http.post("/api/proxy/api/tasks/task-1/answer", async ({ request }) => {
      answerRequests.push(await request.json());
      return HttpResponse.json({
        id: "task-1",
        title: "Launch email",
        spec: "spec",
        status: "WORKING",
        acceptance: "PENDING",
        created_by_type: "USER",
        created_by_id: "user-1",
        owner: maria,
        parent_task_id: null,
        root_task_id: "task-1",
        origin_session_id: "session-1",
        ancestor_expert_ids: ["maria"],
        handoff_count: 0,
        revision_count: 0,
        spend_total: 0,
        outcome_summary: null,
        amendments: [],
        created_at: NOW.toISOString(),
        updated_at: NOW.toISOString(),
        runs: [],
      });
    }),
  );
}

test("answers a task escalation with a one-click option", async () => {
  const user = userEvent.setup();
  const answerRequests: unknown[] = [];
  mockDashboard([escalationItem]);
  mockAnswerEndpoint(answerRequests);

  render(<HomePage />);

  await user.click(await screen.findByRole("button", { name: "Staging" }));

  await waitFor(() => expect(answerRequests).toHaveLength(1));
  expect(answerRequests[0]).toEqual({ answer: "Staging" });
});

test("posts a typed answer to the task escalation", async () => {
  const user = userEvent.setup();
  const answerRequests: unknown[] = [];
  mockDashboard([escalationItem]);
  mockAnswerEndpoint(answerRequests);

  render(<HomePage />);

  const input = await screen.findByRole("textbox", { name: "Your answer" });
  await user.type(input, "Staging first, prod on Friday");
  await user.click(screen.getByRole("button", { name: "Answer" }));

  await waitFor(() => expect(answerRequests).toHaveLength(1));
  expect(answerRequests[0]).toEqual({
    answer: "Staging first, prod on Friday",
  });
});
