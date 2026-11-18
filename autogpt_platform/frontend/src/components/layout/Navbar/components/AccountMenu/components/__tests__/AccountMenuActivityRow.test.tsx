import {
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
import { BackendAPIProvider } from "@/lib/autogpt-server-api/context";
import { server } from "@/mocks/mock-server";
import {
  AFTER_RECONNECT,
  installLiveSocket,
} from "@/tests/integrations/live-socket";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render as renderWithoutProviders } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
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

async function openActivity() {
  render(<AccountMenuActivityRow />);
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
  function Providers({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <BackendAPIProvider>
          <TooltipProvider>{children}</TooltipProvider>
        </BackendAPIProvider>
      </QueryClientProvider>
    );
  }
  // The row mounts only while the account menu is open.
  function AccountMenu({ open }: { open: boolean }) {
    return open ? <AccountMenuActivityRow /> : null;
  }

  const { rerender } = renderWithoutProviders(<AccountMenu open />, {
    wrapper: Providers,
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

test("keeps a run still going that a full list of newer runs leaves out", async () => {
  // GET /executions answers with the newest 250 runs only.
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
  const live = installLiveSocket();

  const list = await openActivity();
  const first = await live.acceptNext();
  await live.publish(first, {
    ...run("run-old", AgentExecutionStatus.RUNNING, "graph-old"),
    started_at: new Date(Date.now() - 2 * 24 * 3_600_000).toISOString(),
  });
  expect(
    (await findRunOf(list, "Old workflow")).getByText(/running$/),
  ).toBeDefined();

  await live.drop(first);
  await live.acceptNext();

  // The refetched list has landed once the new run shows.
  expect(await findRunOf(list, "New workflow", AFTER_RECONNECT)).toBeDefined();
  expect(
    (await findRunOf(list, "Old workflow")).getByText(/running$/),
  ).toBeDefined();
});
