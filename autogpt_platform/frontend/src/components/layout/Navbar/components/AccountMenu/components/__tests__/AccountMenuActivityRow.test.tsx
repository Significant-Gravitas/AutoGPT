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
import { server } from "@/mocks/mock-server";
import { installLiveSocket } from "@/tests/integrations/live-socket";
import { render, screen, within } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
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

afterEach(() => {
  vi.unstubAllGlobals();
});

const GRAPH_ID = "graph-1";

function run(id: string, status: AgentExecutionStatus): GraphExecutionMeta {
  const ended = ![
    AgentExecutionStatus.RUNNING,
    AgentExecutionStatus.QUEUED,
  ].includes(status as "RUNNING" | "QUEUED");
  return {
    ...getGetV1ListAllExecutionsResponseMock()[0],
    id,
    graph_id: GRAPH_ID,
    status,
    started_at: new Date(Date.now() - 60_000),
    ended_at: ended ? new Date() : null,
    is_dry_run: false,
    stats: null,
  };
}

function serveActivity(listRuns: () => Promise<GraphExecutionMeta[]>) {
  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      ...getGetV2ListLibraryAgentsResponseMock(),
      agents: [
        {
          ...getGetV2ListLibraryAgentsResponseMock().agents[0],
          id: "agent-1",
          graph_id: GRAPH_ID,
          name: "Workflow 1",
        },
      ],
      pagination: {
        total_items: 1,
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

async function findRunLine(list: HTMLElement, line: RegExp) {
  return within(list).findByText(line);
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

  expect(await findRunLine(list, /^Completed /)).toBeDefined();
  expect(within(list).queryByText(/running$/)).toBeNull();
});

test("a run's live completion survives a refetch that read the run before it finished", async () => {
  let releaseStaleList = () => {};
  const staleListHeld = new Promise<void>((resolve) => {
    releaseStaleList = resolve;
  });
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
  await vi.waitFor(() => expect(requests).toBe(2));

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

  expect(await findRunLine(list, /running$/)).toBeDefined();
  expect(within(list).queryByText(/^Stopped /)).toBeNull();
});
