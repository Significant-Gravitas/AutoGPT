import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import type { HomeRecentWork } from "@/app/api/__generated__/models/homeRecentWork";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
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
  window_started_at: new Date("2026-08-21T09:00:00Z"),
  completed_count: 3,
  failed_count: 1,
  total_count: 10,
  groups: [
    {
      actor: {
        kind: "expert",
        name: "Maria",
        expert: maria,
        link: "/copilot?expertId=maria",
      },
      latest_at: NOW,
      runs: [
        {
          id: "run-1",
          status: "completed",
          title: "Your camera research is ready",
          summary: "Compared 18 cameras and shortlisted the best three.",
          expert: maria,
          agent_name: "Product Researcher",
          occurred_at: NOW,
          duration_seconds: 812,
          cost_cents: 42,
          link: "/library/agents/lib-2?activeTab=runs&activeItem=run-1",
          trigger: "schedule",
        },
        {
          id: "run-3",
          status: "completed",
          title: "Newsletter draft is ready",
          summary: "Drafted the September newsletter from last week's notes.",
          expert: maria,
          agent_name: "Newsletter Writer",
          occurred_at: NOW,
          duration_seconds: 300,
          cost_cents: 12,
          link: "/library/agents/lib-3?activeTab=runs&activeItem=run-3",
        },
      ],
      items: [
        {
          id: "event-1",
          category: "file",
          event_type: "file.created",
          title: "2026-08-28-code-review-metrics.md",
          occurred_at: NOW,
          file_id: "file-1",
          mime_type: "text/markdown",
          session_title: "Blog pipeline",
          link: "/copilot?sessionId=session-1",
        },
        {
          id: "event-2",
          category: "schedule",
          event_type: "schedule.created",
          title: "persian.sh blog draft",
          occurred_at: NOW,
        },
      ],
      run_count: 3,
      file_count: 2,
      schedule_count: 1,
    },
    {
      actor: {
        kind: "workflow",
        name: "Release Note Generator",
        image_url: "https://example.com/release-notes.png",
        link: "/library/agents/lib-1",
      },
      latest_at: NOW,
      runs: [
        {
          id: "run-2",
          status: "failed",
          title: "Release notes could not be generated",
          summary: "GitHub returned 401 while listing merged pull requests.",
          agent_name: "Release Note Generator",
          occurred_at: NOW,
          duration_seconds: 44,
          cost_cents: 3,
          link: "/library/agents/lib-1?activeTab=runs&activeItem=run-2",
          trigger: "webhook",
        },
      ],
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
      run_count: 1,
      integration_count: 1,
    },
    {
      actor: { kind: "autopilot", name: "AutoPilot", link: "/copilot" },
      latest_at: NOW,
      runs: [],
      items: [
        {
          id: "event-4",
          category: "file",
          event_type: "file.created",
          title: "competitor-pricing.csv",
          occurred_at: NOW,
          link: "/copilot?sessionId=session-2",
        },
      ],
      file_count: 1,
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
    author: { kind: "autopilot", name: "AutoPilot", role: "Head of AI" },
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

test("groups the week's runs and deliverables by who did them", async () => {
  mockDashboard(dashboard);

  render(<HomePage />);

  expect(
    await screen.findByRole("heading", { name: "Recent work" }),
  ).toBeDefined();
  expect(screen.getByText("3 completed")).toBeDefined();
  expect(screen.getByText("1 failed")).toBeDefined();

  const mariaGroup = screen.getByRole("article", { name: "Maria" });
  expect(within(mariaGroup).getByText("Expert")).toBeDefined();
  expect(
    within(mariaGroup).getByText("Your camera research is ready"),
  ).toBeDefined();
  // An expert runs workflows, so the run says which one.
  expect(within(mariaGroup).getByText("Product Researcher")).toBeDefined();
  expect(
    within(mariaGroup).getByText("2026-08-28-code-review-metrics.md"),
  ).toBeDefined();
  expect(within(mariaGroup).getByText("persian.sh blog draft")).toBeDefined();
  expect(
    within(mariaGroup).getByText("3 runs · 2 files · 1 schedule"),
  ).toBeDefined();
  // Every run is one line: the AI summary stays out of the card, and the
  // row says whether a schedule or a person started it.
  expect(
    within(mariaGroup).getByText("Newsletter draft is ready"),
  ).toBeDefined();
  expect(within(mariaGroup).queryByText(/Compared 18 cameras/)).toBeNull();
  expect(
    within(mariaGroup).queryByText(/Drafted the September newsletter/),
  ).toBeNull();
  expect(within(mariaGroup).getByText("Scheduled run")).toBeDefined();
  expect(within(mariaGroup).getByText("Manual run")).toBeDefined();

  const workflowGroup = screen.getByRole("article", {
    name: "Release Note Generator",
  });
  expect(within(workflowGroup).getByText("Workflow")).toBeDefined();
  expect(within(workflowGroup).getByText("1 run · 1 action")).toBeDefined();
  expect(within(workflowGroup).getByText("Failed")).toBeDefined();
  expect(
    within(workflowGroup).getByText("Release notes could not be generated"),
  ).toBeDefined();
  expect(within(workflowGroup).getByText("Triggered run")).toBeDefined();
  expect(within(workflowGroup).queryByText(/GitHub returned 401/)).toBeNull();
  expect(within(workflowGroup).getByText("Send Email")).toBeDefined();
  expect(within(workflowGroup).getByText(/google/)).toBeDefined();

  const autopilotGroup = screen.getByRole("article", { name: "AutoPilot" });
  expect(
    within(autopilotGroup).getByText("competitor-pricing.csv"),
  ).toBeDefined();
});

test("puts the team first and the workflows that ran on their own after them", async () => {
  mockDashboard(dashboard);

  render(<HomePage />);

  const heading = await screen.findByRole("heading", { name: "Recent work" });
  const tile = heading.closest("section");
  if (!tile) throw new Error("Recent work tile not found");
  expect(
    within(tile)
      .getAllByRole("article")
      .map((article) => article.getAttribute("aria-label")),
  ).toEqual(["Maria", "AutoPilot", "Release Note Generator"]);
  // The kind chip says which half a group belongs to, so the rows run
  // straight on without a caption between them.
  expect(within(tile).queryByText("Workflows")).toBeNull();
});

test("marks a workflow group with its own picture and kind", async () => {
  mockDashboard(dashboard);

  render(<HomePage />);

  await screen.findByRole("heading", { name: "Recent work" });
  const workflowGroup = screen.getByRole("article", {
    name: "Release Note Generator",
  });
  expect(within(workflowGroup).getByText("Workflow")).toBeDefined();
  const picture = await within(workflowGroup).findByRole("img", {
    name: "Release Note Generator",
  });
  expect(picture.getAttribute("src")).toBe(
    "https://example.com/release-notes.png",
  );
});

test("links each actor to its home and thread work to its session", async () => {
  mockDashboard(dashboard);

  render(<HomePage />);

  await screen.findByRole("heading", { name: "Recent work" });
  const links = screen
    .getAllByRole("link")
    .map((link) => link.getAttribute("href"));
  expect(links).toContain("/copilot?expertId=maria");
  expect(links).toContain("/library/agents/lib-1");
  expect(links).toContain("/copilot");
  expect(links).toContain("/copilot?sessionId=session-1");
  expect(links).toContain("/copilot?sessionId=session-2");
  expect(
    screen
      .getByRole("link", { name: /code-review-metrics/ })
      .getAttribute("title"),
  ).toBe("Blog pipeline");
});

test("shows a calm empty state when agents produced nothing yet", async () => {
  mockDashboard({
    ...dashboard,
    recent_work: { groups: [], total_count: 0 },
  });

  render(<HomePage />);

  expect(await screen.findByText("Nothing to show yet")).toBeDefined();
  expect(screen.getByText("0 completed")).toBeDefined();
});

test("renders the rest of the page when recent_work is absent", async () => {
  const { recent_work, ...withoutRecentWork } = dashboard;
  void recent_work;
  mockDashboard(withoutRecentWork as HomeDashboardResponse);

  render(<HomePage />);

  expect(await screen.findByText("Nothing to show yet")).toBeDefined();
  expect(screen.getByRole("heading", { name: "Now & next" })).toBeDefined();
});
