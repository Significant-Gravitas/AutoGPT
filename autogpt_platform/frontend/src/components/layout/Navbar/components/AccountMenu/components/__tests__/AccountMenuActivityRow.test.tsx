import { getGetV1GetExecutionDetailsQueryOptions } from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1GetExecutionDetailsResponseMock,
  getGetV1ListAllExecutionsMockHandler,
  getGetV1ListAllExecutionsResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import type { FakeWebSocket } from "@/lib/autogpt-server-api/__tests__/websocket-fake";
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { server } from "@/mocks/mock-server";
import {
  AFTER_RECONNECT,
  installLiveSocket,
} from "@/tests/integrations/live-socket";
import { act, render, screen, within } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render as renderWithoutProviders } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { afterEach, expect, test, vi } from "vitest";
import { AccountMenuActivityRow } from "../AccountMenuActivityRow";

vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));

vi.mock("next/link", () => ({
  __esModule: true,
  default: ({
    children,
    href,
    ...props
  }: React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
  useLinkStatus: () => ({ pending: false }),
}));

let releaseHeldList: (() => void) | undefined;

afterEach(() => {
  // Settle a held response even when the test failed before releasing it.
  releaseHeldList?.();
  releaseHeldList = undefined;
  vi.unstubAllGlobals();
});

const GRAPH_ID = "graph-1";

function run(
  id: string,
  status: AgentExecutionStatus,
  graphID = GRAPH_ID,
): GraphExecutionMeta {
  const ended = ![
    AgentExecutionStatus.RUNNING,
    AgentExecutionStatus.QUEUED,
  ].includes(status as "RUNNING" | "QUEUED");
  return {
    ...getGetV1ListAllExecutionsResponseMock()[0],
    id,
    graph_id: graphID,
    status,
    started_at: new Date(Date.now() - 60_000),
    ended_at: ended ? new Date() : null,
    is_dry_run: false,
    stats: null,
  };
}

function serveActivity(
  listRuns: () => Promise<GraphExecutionMeta[]>,
  workflows: Record<string, string> = { [GRAPH_ID]: "Workflow 1" },
) {
  const agents = Object.entries(workflows).map(([graphID, name]) => ({
    ...getGetV2ListLibraryAgentsResponseMock().agents[0],
    id: `agent-${graphID}`,
    graph_id: graphID,
    name,
  }));
  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      ...getGetV2ListLibraryAgentsResponseMock(),
      agents,
      pagination: {
        total_items: agents.length,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    }),
    getGetV1ListAllExecutionsMockHandler(() => listRuns()),
  );
}

/** The app's providers around a query client the test configures. */
function providersFor(queryClient: QueryClient) {
  return function Providers({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <BackendAPIProvider>
          <TooltipProvider>{children}</TooltipProvider>
        </BackendAPIProvider>
      </QueryClientProvider>
    );
  };
}

async function openActivity(queryClient?: QueryClient) {
  if (queryClient) {
    renderWithoutProviders(<AccountMenuActivityRow />, {
      wrapper: providersFor(queryClient),
    });
  } else {
    render(<AccountMenuActivityRow />);
  }
  await userEvent.click(
    await screen.findByTestId("account-menu-activity-trigger"),
  );
  return screen.findByTestId("agent-activity-dropdown");
}

async function findRunLine(
  list: HTMLElement,
  line: RegExp,
  wait?: typeof AFTER_RECONNECT,
) {
  return within(list).findByText(line, undefined, wait);
}

/** The Activity row of a workflow's run, found by the workflow's name. */
async function findRunOf(
  list: HTMLElement,
  workflow: string,
  wait?: typeof AFTER_RECONNECT,
) {
  const name = await within(list).findByText(workflow, undefined, wait);
  return within(name.closest("a") as HTMLElement);
}

test("a run that finished while the live connection was down shows as finished once it reconnects", async () => {
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  serveActivity(async () => [run("run-1", status)]);
  const live = installLiveSocket();

  const list = await openActivity();
  const first = await live.acceptNext();
  expect(await findRunLine(list, /running$/)).toBeDefined();

  // The list keeps its length: only the run's status changes.
  status = AgentExecutionStatus.COMPLETED;
  await live.drop(first);
  await live.acceptNext();

  expect(await findRunLine(list, /^Completed /, AFTER_RECONNECT)).toBeDefined();
  expect(within(list).queryByText(/running$/)).toBeNull();
});

test("a run's live completion survives a refetch that read the run before it finished", async () => {
  let releaseStaleList = () => {};
  const staleListHeld = new Promise<void>((resolve) => {
    releaseStaleList = resolve;
  });
  releaseHeldList = releaseStaleList;
  let requests = 0;
  serveActivity(async () => {
    requests += 1;
    if (requests === 1) return [run("run-1", AgentExecutionStatus.RUNNING)];
    // Read while run-1 was still going, answered after it finished.
    await staleListHeld;
    return [
      run("run-1", AgentExecutionStatus.RUNNING),
      run("run-2", AgentExecutionStatus.RUNNING),
    ];
  });
  const live = installLiveSocket();

  const list = await openActivity();
  const first = await live.acceptNext();
  expect(await findRunLine(list, /running$/)).toBeDefined();
  await live.drop(first);
  const second = await live.acceptNext();
  await vi.waitFor(() => expect(requests).toBe(2), AFTER_RECONNECT);

  await live.publish(second, {
    ...run("run-1", AgentExecutionStatus.COMPLETED),
    ended_at: new Date().toISOString(),
  });
  expect(await findRunLine(list, /^Completed /)).toBeDefined();
  releaseStaleList();

  // run-2 shows once the stale list has landed.
  expect(await within(list).findAllByText(/running$/)).toHaveLength(1);
  expect(within(list).getByText(/^Completed /)).toBeDefined();
});

test("a refetch that started after a live update shows what it read", async () => {
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  serveActivity(async () => [run("run-1", status)]);
  const live = installLiveSocket();

  const list = await openActivity();
  const first = await live.acceptNext();
  expect(await findRunLine(list, /running$/)).toBeDefined();

  await live.publish(first, {
    ...run("run-1", AgentExecutionStatus.TERMINATED),
    ended_at: new Date().toISOString(),
  });
  expect(await findRunLine(list, /^Stopped /)).toBeDefined();

  // A stopped run can be resumed; a list read after the update is newer.
  status = AgentExecutionStatus.RUNNING;
  await live.drop(first);
  await live.acceptNext();

  expect(await findRunLine(list, /running$/, AFTER_RECONNECT)).toBeDefined();
  expect(within(list).queryByText(/^Stopped /)).toBeNull();
});

test("opening the menu again shows a run that finished while it was closed", async () => {
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  serveActivity(async () => [run("run-1", status)]);
  installLiveSocket();
  // The app keeps a fetched list fresh for a minute
  // (lib/react-query/queryClient.ts).
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 60_000 } },
  });
  // The row mounts only while the account menu is open.
  function AccountMenu({ open }: { open: boolean }) {
    return open ? <AccountMenuActivityRow /> : null;
  }

  const { rerender } = renderWithoutProviders(<AccountMenu open />, {
    wrapper: providersFor(queryClient),
  });
  await userEvent.click(
    await screen.findByTestId("account-menu-activity-trigger"),
  );
  let list = await screen.findByTestId("agent-activity-dropdown");
  expect(await findRunLine(list, /running$/)).toBeDefined();

  rerender(<AccountMenu open={false} />);
  status = AgentExecutionStatus.COMPLETED;
  rerender(<AccountMenu open />);
  await userEvent.click(
    await screen.findByTestId("account-menu-activity-trigger"),
  );
  list = await screen.findByTestId("agent-activity-dropdown");

  expect(await findRunLine(list, /^Completed /)).toBeDefined();
});

const OLD_RUN_STARTED = new Date(
  Date.now() - 2 * 24 * 3_600_000,
).toISOString();

/**
 * GET /executions answers with the newest 250 runs only. After the
 * reconnect, the list leaves out an older run Activity shows as going, and
 * the app reads that run by ID: `readOldRun` answers with its status.
 */
async function showOldRunThenReconnect(
  readOldRun: () => Promise<AgentExecutionStatus>,
  {
    queryClient,
    beforeReconnect,
  }: {
    queryClient?: QueryClient;
    beforeReconnect?: (shown: {
      list: HTMLElement;
      live: ReturnType<typeof installLiveSocket>;
      first: FakeWebSocket;
    }) => Promise<void>;
  } = {},
) {
  const newer = Array.from({ length: 250 }, (_, i) =>
    run(`run-${i}`, AgentExecutionStatus.COMPLETED, "graph-busy"),
  );
  let requests = 0;
  serveActivity(
    async () => {
      requests += 1;
      if (requests === 1) return newer;
      return [
        run("run-new", AgentExecutionStatus.RUNNING, "graph-new"),
        ...newer.slice(0, 249),
      ];
    },
    {
      "graph-old": "Old workflow",
      "graph-busy": "Busy workflow",
      "graph-new": "New workflow",
    },
  );
  server.use(
    getGetV1GetExecutionDetailsMockHandler(async () => ({
      ...getGetV1GetExecutionDetailsResponseMock(),
      ...run("run-old", await readOldRun(), "graph-old"),
      inputs: {},
      outputs: {},
      node_executions: [],
    })),
  );
  const live = installLiveSocket();

  const list = await openActivity(queryClient);
  const first = await live.acceptNext();
  await live.publish(first, {
    ...run("run-old", AgentExecutionStatus.RUNNING, "graph-old"),
    started_at: OLD_RUN_STARTED,
  });
  expect(
    (await findRunOf(list, "Old workflow")).getByText(/running$/),
  ).toBeDefined();

  await beforeReconnect?.({ list, live, first });
  await live.drop(first);
  const second = await live.acceptNext();
  // The refetched list has landed once the new run shows.
  expect(await findRunOf(list, "New workflow", AFTER_RECONNECT)).toBeDefined();
  return { list, live, second };
}

test("keeps a run still going that a full list of newer runs leaves out", async () => {
  const { list } = await showOldRunThenReconnect(
    async () => AgentExecutionStatus.RUNNING,
  );

  expect(
    (await findRunOf(list, "Old workflow")).getByText(/running$/),
  ).toBeDefined();
});

test("a run a full list leaves out that ended while the live connection was down shows as ended", async () => {
  const { list } = await showOldRunThenReconnect(
    async () => AgentExecutionStatus.COMPLETED,
  );

  const oldRun = await findRunOf(list, "Old workflow");
  expect(await oldRun.findByText(/^Completed /)).toBeDefined();
  expect(oldRun.queryByText(/running$/)).toBeNull();
});

test("a live update that lands while a left-out run is read by ID stands", async () => {
  let readStarted = false;
  let releaseRead = () => {};
  const readHeld = new Promise<void>((resolve) => {
    releaseRead = resolve;
  });
  releaseHeldList = releaseRead;
  const { list, live, second } = await showOldRunThenReconnect(async () => {
    readStarted = true;
    // Read before the run was stopped, answered after.
    await readHeld;
    return AgentExecutionStatus.RUNNING;
  });
  await vi.waitFor(() => expect(readStarted).toBe(true));

  await live.publish(second, {
    ...run("run-old", AgentExecutionStatus.TERMINATED, "graph-old"),
    started_at: OLD_RUN_STARTED,
  });
  const oldRun = await findRunOf(list, "Old workflow");
  expect(await oldRun.findByText(/^Stopped /)).toBeDefined();
  releaseRead();
  await act(() => new Promise((resolve) => setTimeout(resolve, 200)));

  expect(
    (await findRunOf(list, "Old workflow")).getByText(/^Stopped /),
  ).toBeDefined();
});

test("a left-out run's read by ID doesn't take an answer that was already on its way", async () => {
  // The run's own view polls the same query. Its read goes out before the
  // run stops for review and comes back after Activity's check has started.
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  let reads = 0;
  let releasePoll = () => {};
  const pollHeld = new Promise<void>((resolve) => {
    releasePoll = resolve;
  });
  releaseHeldList = releasePoll;
  const { list } = await showOldRunThenReconnect(
    async () => {
      reads += 1;
      if (reads > 1) return AgentExecutionStatus.REVIEW;
      await pollHeld;
      return AgentExecutionStatus.RUNNING;
    },
    {
      queryClient,
      async beforeReconnect({ list, live, first }) {
        void queryClient
          .fetchQuery(
            getGetV1GetExecutionDetailsQueryOptions("graph-old", "run-old"),
          )
          .catch(() => undefined);
        await vi.waitFor(() => expect(reads).toBe(1));
        await live.publish(first, {
          ...run("run-old", AgentExecutionStatus.REVIEW, "graph-old"),
          started_at: OLD_RUN_STARTED,
          ended_at: null,
        });
        expect(
          await (
            await findRunOf(list, "Old workflow")
          ).findByText(/^Awaiting approval /),
        ).toBeDefined();
      },
    },
  );
  releasePoll();
  await act(() => new Promise((resolve) => setTimeout(resolve, 200)));

  const oldRun = await findRunOf(list, "Old workflow");
  expect(oldRun.getByText(/^Awaiting approval /)).toBeDefined();
  expect(oldRun.queryByText(/running$/)).toBeNull();
});

test("a retried refetch isn't overruled by an update from before its first attempt failed", async () => {
  let releaseFailure = () => {};
  const failureHeld = new Promise<void>((resolve) => {
    releaseFailure = resolve;
  });
  releaseHeldList = releaseFailure;
  serveActivity(async () => [run("run-1", AgentExecutionStatus.RUNNING)]);
  let requests = 0;
  server.use(
    http.get("/api/proxy/api/executions", async () => {
      requests += 1;
      if (requests !== 2) {
        // Stopped, then resumed before the retry read it.
        return HttpResponse.json([run("run-1", AgentExecutionStatus.RUNNING)]);
      }
      await failureHeld;
      return HttpResponse.json({ detail: "Unavailable" }, { status: 503 });
    }),
  );
  const live = installLiveSocket();
  // The app retries a failed read (lib/react-query/queryClient.ts).
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: 1, retryDelay: 0 } },
  });
  renderWithoutProviders(<AccountMenuActivityRow />, {
    wrapper: providersFor(queryClient),
  });
  await userEvent.click(
    await screen.findByTestId("account-menu-activity-trigger"),
  );
  const list = await screen.findByTestId("agent-activity-dropdown");
  const first = await live.acceptNext();
  expect(await findRunLine(list, /running$/)).toBeDefined();

  await live.drop(first);
  const second = await live.acceptNext();
  await vi.waitFor(() => expect(requests).toBe(2), AFTER_RECONNECT);
  // The stop arrives while the reconnect read is out, and that read fails.
  await live.publish(second, {
    ...run("run-1", AgentExecutionStatus.TERMINATED),
    ended_at: new Date().toISOString(),
  });
  expect(await findRunLine(list, /^Stopped /)).toBeDefined();
  releaseFailure();

  await vi.waitFor(() => expect(requests).toBe(3));
  expect(await findRunLine(list, /running$/)).toBeDefined();
  expect(within(list).queryByText(/^Stopped /)).toBeNull();
});
