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
import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { server } from "@/mocks/mock-server";
import {
  AFTER_RECONNECT,
  installLiveSocket,
} from "@/tests/integrations/live-socket";
import { act, render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, expect, test, vi } from "vitest";
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

afterEach(() => {
  vi.unstubAllGlobals();
});

const GRAPH_ID = "graph-1";

interface Served {
  status: () => AgentExecutionStatus;
  dryRun?: boolean;
}

function serveAgentPage({ status: runStatus, dryRun = false }: Served) {
  function run() {
    const status = runStatus();
    return {
      id: "run-1",
      graph_id: GRAPH_ID,
      graph_version: 1,
      status,
      started_at: new Date(Date.now() - 60_000),
      ended_at: status === AgentExecutionStatus.RUNNING ? null : new Date(),
      is_dry_run: dryRun,
      stats: null,
    };
  }
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: GRAPH_ID,
        graph_version: 1,
        name: "Reconnect workflow",
      }),
    ),
    getGetV1ListGraphExecutionsMockHandler(() => ({
      ...getGetV1ListGraphExecutionsResponseMock(),
      executions: [
        {
          ...getGetV1ListGraphExecutionsResponseMock().executions[0],
          ...run(),
        },
      ],
      pagination: {
        total_items: 1,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    })),
    getGetV1GetExecutionDetailsMockHandler(() => ({
      ...getGetV1GetExecutionDetailsResponseMock(),
      ...run(),
      inputs: {},
      outputs: {},
      node_executions: [],
    })),
    getGetV1ListExecutionSchedulesForAGraphMockHandler([]),
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

test("a run that finished while the live connection was down shows as finished in the runs list once it reconnects", async () => {
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  serveAgentPage({ status: () => status });
  const live = installLiveSocket();

  render(
    <NuqsTestingAdapter hasMemory searchParams="activeTab=runs">
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );
  const first = await live.acceptNext();
  expect(
    await screen.findByRole("img", { name: "Task running" }),
  ).toBeDefined();

  status = AgentExecutionStatus.COMPLETED;
  await live.drop(first);
  await live.acceptNext();

  expect(
    await screen.findByRole("img", { name: "Task completed" }, AFTER_RECONNECT),
  ).toBeDefined();
  expect(screen.queryByRole("img", { name: "Task running" })).toBeNull();
});

test("a run that finished during a disconnect on another tab shows as finished back on Tasks", async () => {
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  serveAgentPage({ status: () => status });
  const live = installLiveSocket();

  render(
    <NuqsTestingAdapter hasMemory searchParams="activeTab=scheduled">
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );
  const first = await live.acceptNext();
  expect(
    await screen.findByRole("tab", { name: /^Scheduled/, selected: true }),
  ).toBeDefined();

  let runReads = 0;
  function countRunReads({ request }: { request: Request }) {
    const { pathname } = new URL(request.url);
    if (pathname.endsWith(`/graphs/${GRAPH_ID}/executions`)) runReads += 1;
  }
  server.events.on("request:start", countRunReads);
  try {
    status = AgentExecutionStatus.COMPLETED;
    await live.drop(first);
    await live.caughtUp(await live.acceptNext());
    // Scheduled doesn't show the runs, so they wait for Tasks: a failed read
    // here would only put an error in front of the schedules.
    await act(() => new Promise((resolve) => setTimeout(resolve, 200)));
    expect(runReads).toBe(0);

    await userEvent.click(screen.getByRole("tab", { name: /^Tasks/ }));
    expect(
      await screen.findByRole("img", { name: "Task completed" }),
    ).toBeDefined();
    expect(runReads).toBeGreaterThan(0);
  } finally {
    server.events.removeListener("request:start", countRunReads);
  }
});

test("a simulated run's icon names its status", async () => {
  serveAgentPage({ status: () => AgentExecutionStatus.RUNNING, dryRun: true });
  installLiveSocket();

  render(
    <NuqsTestingAdapter hasMemory searchParams="activeTab=runs">
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );

  expect(
    await screen.findByRole("img", { name: "Simulated task running" }),
  ).toBeDefined();
});
