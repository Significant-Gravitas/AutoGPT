import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import type { HomeRecentWork } from "@/app/api/__generated__/models/homeRecentWork";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
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
  useAuth: () => ({
    user: { user_metadata: { preferred_name: "Ubbe" } },
  }),
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

const NOW = new Date("2026-08-28T09:00:00Z");
const maria = {
  id: "maria",
  name: "Maria",
  role: "Marketing",
  avatar_url: null,
};

const recentWork: HomeRecentWork = {
  total_count: 10,
  groups: [
    {
      actor: { kind: "expert", name: "Maria", expert: maria },
      session_id: "session-1",
      session_title: "Blog pipeline",
      link: "/copilot?sessionId=session-1",
      latest_at: NOW,
      items: [
        {
          id: "event-1",
          category: "file",
          event_type: "file.created",
          title: "2026-08-28-code-review-metrics.md",
          occurred_at: NOW,
          file_id: "file-1",
          mime_type: "text/markdown",
        },
        {
          id: "event-2",
          category: "schedule",
          event_type: "schedule.created",
          title: "persian.sh blog draft",
          occurred_at: NOW,
        },
      ],
      more_count: 6,
    },
    {
      actor: { kind: "autopilot", name: "Autopilot" },
      session_id: "session-2",
      latest_at: NOW,
      items: [
        {
          id: "event-3",
          category: "integration",
          event_type: "integration.action",
          title: "Send Email",
          occurred_at: NOW,
          provider: "google",
        },
      ],
      more_count: 0,
    },
  ],
};

const dashboard: HomeDashboardResponse = {
  generated_at: NOW,
  timezone: "UTC",
  attention: [],
  briefing: {
    generated_at: NOW,
    window_started_at: NOW,
    completed_count: 0,
    failed_count: 0,
    routine_count: 0,
    outcomes: [],
  },
  active_tasks: [],
  upcoming_tasks: [],
  team: { total: 1, ready: 1, working: 0, needs_attention: 0 },
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
  recent_work: recentWork,
};

function mockDashboard(response: HomeDashboardResponse) {
  server.use(
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(response),
    ),
  );
}

test("shows agent work grouped by actor and thread", async () => {
  mockDashboard(dashboard);

  render(<HomePage />);

  expect(
    await screen.findByRole("heading", { name: "Recent work" }),
  ).toBeDefined();
  expect(screen.getByText("Maria")).toBeDefined();
  expect(screen.getByText("Blog pipeline")).toBeDefined();
  expect(screen.getByText("2026-08-28-code-review-metrics.md")).toBeDefined();
  expect(screen.getByText("persian.sh blog draft")).toBeDefined();
  expect(screen.getByText("Plus 6 more")).toBeDefined();
  expect(screen.getByText("Autopilot")).toBeDefined();
  expect(screen.getByText("Send Email")).toBeDefined();
  expect(screen.getByText(/google/)).toBeDefined();
});

test("links a thread-backed group to its copilot session", async () => {
  mockDashboard(dashboard);

  render(<HomePage />);

  await screen.findByRole("heading", { name: "Recent work" });
  const links = screen
    .getAllByRole("link")
    .map((link) => link.getAttribute("href"));
  expect(links).toContain("/copilot?sessionId=session-1");
});

test("shows a calm empty state when agents produced nothing yet", async () => {
  mockDashboard({
    ...dashboard,
    recent_work: { groups: [], total_count: 0 },
  });

  render(<HomePage />);

  expect(await screen.findByText("Nothing to show yet")).toBeDefined();
});

test("renders the rest of the page when recent_work is absent", async () => {
  const { recent_work, ...withoutRecentWork } = dashboard;
  void recent_work;
  mockDashboard(withoutRecentWork as HomeDashboardResponse);

  render(<HomePage />);

  expect(await screen.findByText("Nothing to show yet")).toBeDefined();
  expect(screen.getByRole("heading", { name: "Now & next" })).toBeDefined();
});
