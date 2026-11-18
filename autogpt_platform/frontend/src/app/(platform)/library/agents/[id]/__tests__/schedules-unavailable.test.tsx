import {
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1GetExecutionDetailsResponseMock,
  getGetV1ListGraphExecutionsMockHandler,
  getGetV1ListGraphExecutionsResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListPresetsMockHandler } from "@/app/api/__generated__/endpoints/presets/presets.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { server } from "@/mocks/mock-server";
import { installLiveSocket } from "@/tests/integrations/live-socket";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { expect, onTestFinished, test, vi } from "vitest";
import AgentLibraryPage from "../page";

vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useParams: () => ({ id: "library-agent" }),
  usePathname: () => "/library/agents/library-agent",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "u@example.com" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => false,
  useFlagStatus: () => ({ enabled: false, ready: true }),
}));

const GRAPH_ID = "graph-1";
const NOTICE =
  "Schedules are temporarily unavailable — your schedules are safe and will run once it's back.";

// What the backend answers while the scheduler restarts on a deploy.
function schedulerOut() {
  return HttpResponse.json(
    {
      message: `Failed to process GET /api/graphs/${GRAPH_ID}/schedules`,
      detail: "Schedules are temporarily unavailable. Try again in a moment.",
      hint: "Retry in 15 seconds.",
    },
    { status: 503, headers: { "Retry-After": "15" } },
  );
}

const morningRun = {
  id: "schedule-1",
  name: "Every morning",
  user_id: "user-1",
  graph_id: GRAPH_ID,
  graph_version: 1,
  cron: "0 8 * * *",
  timezone: "UTC",
  next_run_time: new Date(Date.now() + 60 * 60 * 1000).toISOString(),
  input_data: {},
  input_credentials: {},
};

function serveAgentPage(
  schedules: () => Response | Promise<Response>,
  { withRuns = true }: { withRuns?: boolean } = {},
) {
  const run = {
    ...getGetV1ListGraphExecutionsResponseMock().executions[0],
    id: "run-1",
    graph_id: GRAPH_ID,
    graph_version: 1,
    status: AgentExecutionStatus.COMPLETED,
    started_at: new Date(Date.now() - 120_000),
    ended_at: new Date(Date.now() - 60_000),
    is_dry_run: false,
    stats: null,
  };
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: GRAPH_ID,
        graph_version: 1,
        name: "Morning digest",
      }),
    ),
    getGetV1ListGraphExecutionsMockHandler({
      ...getGetV1ListGraphExecutionsResponseMock(),
      executions: withRuns ? [run] : [],
      pagination: {
        total_items: withRuns ? 1 : 0,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    }),
    getGetV1GetExecutionDetailsMockHandler(() => ({
      ...getGetV1GetExecutionDetailsResponseMock(),
      ...run,
      inputs: {},
      outputs: {},
      node_executions: [],
    })),
    http.get("*/api/graphs/:graphId/schedules", () => schedules()),
    getGetV2ListPresetsMockHandler({
      presets: [],
      pagination: {
        total_items: 0,
        total_pages: 0,
        current_page: 1,
        page_size: 100,
      },
    }),
  );
}

test("a scheduler outage keeps the workflow's tasks and says schedules are unavailable, then recovers", async () => {
  let schedulerUp = false;
  serveAgentPage(() =>
    schedulerUp ? HttpResponse.json([morningRun]) : schedulerOut(),
  );
  installLiveSocket();

  render(
    <NuqsTestingAdapter hasMemory searchParams="activeTab=runs">
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );

  // The tasks still list, and nothing stands in for them as an error.
  expect(
    await screen.findByRole("img", { name: "Task completed" }),
  ).toBeDefined();
  expect(screen.queryByText("Something went wrong")).toBeNull();
  expect(
    screen
      .getByRole("button", { name: /New workflow task/ })
      .hasAttribute("disabled"),
  ).toBe(false);

  // The schedule count is unknown, and the tab says why it lists nothing.
  const scheduledTab = screen.getByRole("tab", { name: /^Scheduled/ });
  expect(within(scheduledTab).getByText("—")).toBeDefined();
  await userEvent.click(scheduledTab);
  expect(await screen.findAllByText(NOTICE)).not.toHaveLength(0);
  expect(screen.queryByText("No scheduled workflows")).toBeNull();
  expect(screen.queryByText("Nothing scheduled yet")).toBeNull();

  schedulerUp = true;
  fireEvent.click(screen.getAllByRole("button", { name: "Try again" })[0]);

  // Listed in the sidebar, and opened beside it.
  expect(await screen.findAllByText("Every morning")).not.toHaveLength(0);
  await waitFor(() => expect(screen.queryByText(NOTICE)).toBeNull());
  expect(
    within(screen.getByRole("tab", { name: /^Scheduled/ })).getByText("1"),
  ).toBeDefined();
  // The schedule that came back is opened, not "Nothing scheduled yet".
  expect(await screen.findByText("Next run")).toBeDefined();
  expect(screen.queryByText("Nothing scheduled yet")).toBeNull();
});

test("a recheck still waiting on the scheduler keeps the tabs and the notice", async () => {
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  onTestFinished(() => release());
  let reads = 0;
  serveAgentPage(async () => {
    reads += 1;
    if (reads === 1) return schedulerOut();
    await held;
    return schedulerOut();
  });
  installLiveSocket();

  render(
    <NuqsTestingAdapter hasMemory searchParams="activeTab=scheduled">
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );
  expect(await screen.findAllByText(NOTICE)).not.toHaveLength(0);

  fireEvent.click(screen.getByRole("button", { name: "Try again" }));
  await waitFor(() => expect(reads).toBe(2));

  // A schedule list that never loaded drops its error while it refetches;
  // the page still knows the scheduler is out.
  expect(screen.getByRole("tab", { name: /^Tasks/ })).toBeDefined();
  expect(screen.getAllByText(NOTICE)).not.toHaveLength(0);

  release();
  // Let the held answer land before the test ends.
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 50));
  });
  expect(screen.getAllByText(NOTICE)).not.toHaveLength(0);
});

test("a workflow with no tasks yet isn't shown as empty while its schedules are unknown", async () => {
  serveAgentPage(() => schedulerOut(), { withRuns: false });
  installLiveSocket();

  render(
    <NuqsTestingAdapter hasMemory searchParams="activeTab=runs">
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );

  // It may have schedules, so the page keeps its tabs instead of offering a
  // first run.
  const scheduledTab = await screen.findByRole("tab", { name: /^Scheduled/ });
  expect(within(scheduledTab).getByText("—")).toBeDefined();
  await userEvent.click(scheduledTab);
  expect(await screen.findAllByText(NOTICE)).not.toHaveLength(0);
  expect(screen.queryByText("Nothing scheduled yet")).toBeNull();
});

test("a schedule open on the page says it's unavailable rather than failing, and comes back", async () => {
  let schedulerUp = false;
  serveAgentPage(() =>
    schedulerUp ? HttpResponse.json([morningRun]) : schedulerOut(),
  );
  installLiveSocket();

  render(
    <NuqsTestingAdapter
      hasMemory
      searchParams={`activeTab=scheduled&activeItem=${morningRun.id}`}
    >
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );

  // Both the list and the open schedule say why, and neither is an error.
  await waitFor(() =>
    expect(screen.getAllByText(NOTICE).length).toBeGreaterThan(1),
  );
  expect(screen.queryByText("Something went wrong")).toBeNull();

  schedulerUp = true;
  fireEvent.click(screen.getByRole("button", { name: "Try again" }));

  expect(await screen.findByText("Next run")).toBeDefined();
  await waitFor(() => expect(screen.queryByText(NOTICE)).toBeNull());
});
