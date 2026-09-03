import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
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
  recent_work: { groups: [], total_count: 0 },
};

function mockDashboard(response: HomeDashboardResponse) {
  server.use(
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(response),
    ),
  );
}

// The workflow rows read the library's own agent and execution feeds, not
// the home aggregate, so those are what decide whether they show.
function mockLibraryAgents(agents: Array<Partial<LibraryAgent>>) {
  const base = getGetV2ListLibraryAgentsResponseMock();
  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      ...base,
      agents: agents.map((agent, index) => ({
        ...base.agents[0],
        id: `agent-${index}`,
        graph_id: `graph-${index}`,
        has_external_trigger: false,
        is_scheduled: false,
        next_scheduled_run: null,
        ...agent,
      })),
      pagination: {
        total_items: agents.length,
        total_pages: 1,
        current_page: 1,
        page_size: 100,
      },
    }),
    getGetV1ListAllExecutionsMockHandler([]),
  );
}

test("lists each workflow's latest state under Recent work", async () => {
  mockDashboard(dashboard);
  const inTwoDays = new Date(Date.now() + 2 * 24 * 60 * 60 * 1000);
  mockLibraryAgents([
    { name: "Inbox Watcher", has_external_trigger: true },
    {
      name: "Daily Blog Draft",
      is_scheduled: true,
      next_scheduled_run: inTwoDays.toISOString(),
    },
  ]);

  render(<HomePage />);

  const workflows = await screen.findByRole("region", { name: "Workflows" });
  const card = screen
    .getByRole("heading", { name: "Recent work" })
    .closest("section");
  expect(card?.contains(workflows)).toBe(true);
  expect(within(workflows).getByText("Inbox Watcher")).toBeDefined();
  expect(
    within(workflows).getByText("Waiting for trigger event"),
  ).toBeDefined();
  expect(
    within(workflows).getByRole("link", { name: /See/ }).getAttribute("href"),
  ).toBe("/library/agents/agent-0");
  // Scheduled runs belong to Now & next's Coming up, so they do not repeat.
  expect(within(workflows).queryByText("Daily Blog Draft")).toBeNull();
  // Rows are content, so the card's empty state stands down.
  expect(screen.queryByText("Nothing to show yet")).toBeNull();
});

test("keeps the empty state when only scheduled workflows exist", async () => {
  mockDashboard(dashboard);
  mockLibraryAgents([
    {
      name: "Daily Blog Draft",
      is_scheduled: true,
      next_scheduled_run: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
    },
  ]);

  render(<HomePage />);

  expect(await screen.findByText("Nothing to show yet")).toBeDefined();
  expect(screen.queryByRole("region", { name: "Workflows" })).toBeNull();
});
