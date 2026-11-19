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
import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { QueryClient } from "@tanstack/react-query";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { afterEach, expect, onTestFinished, test, vi } from "vitest";
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

let releaseHeldRuns: (() => void) | undefined;

afterEach(() => {
  // Settle a held response even when the test failed before releasing it.
  releaseHeldRuns?.();
  releaseHeldRuns = undefined;
  vi.unstubAllGlobals();
});

const GRAPH_ID = "graph-1";

interface Served {
  status: () => AgentExecutionStatus;
  dryRun?: boolean;
  /** Holds the answer to the first read of the runs list until it settles. */
  holdFirstRuns?: Promise<void>;
}

function serveAgentPage({
  status: runStatus,
  dryRun = false,
  holdFirstRuns,
}: Served) {
  let runReads = 0;
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
    getGetV1ListGraphExecutionsMockHandler(async () => {
      runReads += 1;
      const read = run();
      if (runReads === 1 && holdFirstRuns) await holdFirstRuns;
      return {
        ...getGetV1ListGraphExecutionsResponseMock(),
        executions: [
          {
            ...getGetV1ListGraphExecutionsResponseMock().executions[0],
            ...read,
          },
        ],
        pagination: {
          total_items: 1,
          total_pages: 1,
          current_page: 1,
          page_size: 20,
        },
      };
    }),
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
  return { runReads: () => runReads };
}

test("the runs list reads its runs once on a first load", async () => {
  const { runReads } = serveAgentPage({
    status: () => AgentExecutionStatus.RUNNING,
  });
  installLiveSocket();
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  render(
    <NuqsTestingAdapter hasMemory searchParams="activeTab=runs">
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
    { queryClient },
  );
  expect(
    await screen.findByRole("img", { name: "Task running" }),
  ).toBeDefined();
  // Let any read the first render started get under way.
  await act(() => new Promise((resolve) => setTimeout(resolve, 0)));

  const [runs] = queryClient.getQueryCache().findAll({
    predicate: (query) =>
      String(query.queryKey[0]).endsWith(`/graphs/${GRAPH_ID}/executions`),
  });
  expect(runs?.state).toMatchObject({
    fetchStatus: "idle",
    isInvalidated: false,
  });
  expect(runReads()).toBe(1);
});

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
  const { runReads } = serveAgentPage({ status: () => status });
  const live = installLiveSocket();
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  render(
    <NuqsTestingAdapter hasMemory searchParams="activeTab=scheduled">
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
    { queryClient },
  );
  const first = await live.acceptNext();
  expect(
    await screen.findByRole("tab", { name: /^Scheduled/, selected: true }),
  ).toBeDefined();
  // The runs list reads once on its own, whichever tab is open.
  await waitFor(() => expect(runReads()).toBe(1));
  const [runs] = queryClient.getQueryCache().findAll({
    predicate: (query) =>
      String(query.queryKey[0]).endsWith(`/graphs/${GRAPH_ID}/executions`),
  });
  await waitFor(() => expect(runs?.state.fetchStatus).toBe("idle"));
  const readAt = runs?.state.dataUpdatedAt;

  // No spread before the reconnect notice, so any read the reconnect
  // schedules is under way within the tick below.
  const random = vi.spyOn(Math, "random").mockReturnValue(0);
  onTestFinished(() => random.mockRestore());
  status = AgentExecutionStatus.COMPLETED;
  await live.drop(first);
  await live.acceptNext();
  await live.caughtUp();
  await act(() => new Promise((resolve) => setTimeout(resolve, 0)));
  // Scheduled doesn't show the runs, so they wait for Tasks: a failed read
  // here would only put an error in front of the schedules.
  expect(runs?.state).toMatchObject({
    fetchStatus: "idle",
    isInvalidated: false,
    dataUpdatedAt: readAt,
  });

  await userEvent.click(screen.getByRole("tab", { name: /^Tasks/ }));
  expect(
    await screen.findByRole("img", { name: "Task completed" }),
  ).toBeDefined();
  expect(runReads()).toBeGreaterThan(1);
});

test("runs still loading on Tasks when the live connection drops are read again once it reconnects", async () => {
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  const { runReads } = serveAgentPage({
    status: () => status,
    holdFirstRuns: new Promise<void>((resolve) => {
      releaseHeldRuns = resolve;
    }),
  });
  const live = installLiveSocket();

  render(
    <NuqsTestingAdapter hasMemory searchParams="activeTab=runs">
      <AgentLibraryPage />
    </NuqsTestingAdapter>,
  );
  const first = await live.acceptNext();
  // The first read went out while the run was going; its answer is held.
  await waitFor(() => expect(runReads()).toBe(1));

  status = AgentExecutionStatus.COMPLETED;
  await live.drop(first);
  await live.acceptNext();

  expect(
    await screen.findByRole("img", { name: "Task completed" }, AFTER_RECONNECT),
  ).toBeDefined();
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
