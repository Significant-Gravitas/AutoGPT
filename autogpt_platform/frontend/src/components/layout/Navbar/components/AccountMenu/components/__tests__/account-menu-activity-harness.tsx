import { getV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
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
import { expect, vi } from "vitest";
import { AccountMenuActivityRow } from "../AccountMenuActivityRow";

export const GRAPH_ID = "graph-1";

export function run(
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

export function serveActivity(
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
export function providersFor(queryClient: QueryClient) {
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

export async function openActivity(queryClient?: QueryClient) {
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

export async function findRunLine(
  list: HTMLElement,
  line: RegExp,
  wait?: typeof AFTER_RECONNECT,
) {
  return within(list).findByText(line, undefined, wait);
}

/** Waits until Activity has taken the answer to this test's latest read by ID. */
export async function latestReadTaken() {
  const read = vi.mocked(getV1GetExecutionDetails).mock.results.at(-1);
  // Cleared before each test, so a test whose read never happened fails here.
  expect(read).toBeDefined();
  // Activity awaited this promise first, so its handling has run by now.
  await Promise.resolve(read?.value).catch(() => undefined);
  await act(async () => {});
}

/** The Activity row of a workflow's run, found by the workflow's name. */
export async function findRunOf(
  list: HTMLElement,
  workflow: string,
  wait?: typeof AFTER_RECONNECT,
) {
  const name = await within(list).findByText(workflow, undefined, wait);
  return within(name.closest("a") as HTMLElement);
}

export const OLD_RUN_STARTED = new Date(
  Date.now() - 2 * 24 * 3_600_000,
).toISOString();

/**
 * GET /executions answers with the newest 250 runs only. After the
 * reconnect, the list leaves out an older run Activity shows as going, and
 * the app reads that run by ID: `readOldRun` answers with its status, or
 * with an HTTP error status.
 */
export async function showOldRunThenReconnect(
  readOldRun: () => Promise<AgentExecutionStatus | number>,
  {
    queryClient,
    beforeReconnect,
    listedAgainFrom,
  }: {
    queryClient?: QueryClient;
    beforeReconnect?: (shown: {
      list: HTMLElement;
      live: ReturnType<typeof installLiveSocket>;
      first: FakeWebSocket;
    }) => Promise<void>;
    /** From this list request on, the old run is in the list, completed. */
    listedAgainFrom?: number;
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
      const recent = [
        run("run-new", AgentExecutionStatus.RUNNING, "graph-new"),
        ...newer.slice(0, 249),
      ];
      if (listedAgainFrom === undefined || requests < listedAgainFrom) {
        return recent;
      }
      return [
        {
          ...run("run-old", AgentExecutionStatus.COMPLETED, "graph-old"),
          started_at: new Date(OLD_RUN_STARTED),
        },
        ...recent.slice(0, 249),
      ];
    },
    {
      "graph-old": "Old workflow",
      "graph-busy": "Busy workflow",
      "graph-new": "New workflow",
    },
  );
  server.use(
    http.get(
      "/api/proxy/api/graphs/:graphId/executions/:graphExecId",
      async () => {
        const read = await readOldRun();
        if (typeof read === "number") {
          return HttpResponse.json({ detail: "Read failed" }, { status: read });
        }
        return HttpResponse.json({
          ...getGetV1GetExecutionDetailsResponseMock(),
          ...run("run-old", read, "graph-old"),
          inputs: {},
          outputs: {},
          node_executions: [],
        });
      },
    ),
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
