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
