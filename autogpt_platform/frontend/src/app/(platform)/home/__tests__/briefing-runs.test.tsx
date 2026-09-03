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

// The runs strip reads the library's own agent and execution feeds, not the
// home aggregate, so those are what decide whether it shows.
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

test("surfaces recent workflow runs under the briefing", async () => {
  mockDashboard(dashboard);
  mockLibraryAgents([{ name: "Inbox Watcher", has_external_trigger: true }]);

  render(<HomePage />);

  const runs = await screen.findByRole("region", {
    name: "Recent workflow runs",
  });
  const briefing = screen
    .getByRole("heading", { name: "Your briefing" })
    .closest("section");
  expect(briefing?.contains(runs)).toBe(true);
  expect(within(runs).getByText("Inbox Watcher")).toBeDefined();
  expect(within(runs).getByText("Waiting for trigger event")).toBeDefined();
  expect(
    within(runs).getByRole("link", { name: /See/ }).getAttribute("href"),
  ).toBe("/library/agents/agent-0");
  expect(
    within(runs).getByRole("link", { name: /Ask/ }).getAttribute("href"),
  ).toContain("/copilot?autosubmit=true#prompt=");
});

test("keeps the briefing's empty state clean when no workflow has run", async () => {
  mockDashboard(dashboard);
  mockLibraryAgents([]);

  render(<HomePage />);

  expect(await screen.findByText("No new outcomes yet")).toBeDefined();
  expect(
    screen.queryByRole("region", { name: "Recent workflow runs" }),
  ).toBeNull();
});
