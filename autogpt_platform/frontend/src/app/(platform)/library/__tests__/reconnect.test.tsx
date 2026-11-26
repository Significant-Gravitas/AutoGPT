import { afterEach, expect, onTestFinished, test, vi } from "vitest";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  AFTER_RECONNECT,
  installLiveSocket,
  type SentMessage,
} from "@/tests/integrations/live-socket";
import type { FakeWebSocket } from "@/lib/autogpt-server-api/__tests__/websocket-fake";
import {
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import {
  getGetV2ListLibraryFoldersMockHandler,
  getGetV2ListLibraryFoldersResponseMock,
} from "@/app/api/__generated__/endpoints/folders/folders.msw";
import {
  getGetV1ListAllExecutionsMockHandler,
  getGetV1ListAllExecutionsResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { AccountMenuActivityRow } from "@/components/layout/Navbar/components/AccountMenu/components/AccountMenuActivityRow";
import LibraryPage from "../page";

vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));

let releaseHeldList: (() => void) | undefined;

afterEach(() => {
  // Settle a held response even when the test failed before releasing it.
  releaseHeldList?.();
  releaseHeldList = undefined;
  vi.unstubAllGlobals();
});

const agent = {
  ...getGetV2ListLibraryAgentsResponseMock().agents[0],
  id: "agent-1",
  graph_id: "graph-1",
  name: "Workflow 1",
};

function serveLibrary(
  runStatus: () => AgentExecutionStatus | Promise<AgentExecutionStatus>,
) {
  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      ...getGetV2ListLibraryAgentsResponseMock(),
      agents: [agent],
      pagination: {
        total_items: 1,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    }),
    getGetV2ListLibraryFoldersMockHandler(
      getGetV2ListLibraryFoldersResponseMock({
        folders: [],
        pagination: {
          total_items: 0,
          total_pages: 1,
          current_page: 1,
          page_size: 20,
        },
      }),
    ),
    getGetV1ListAllExecutionsMockHandler(async () => {
      const status = await runStatus();
      return [
        {
          ...getGetV1ListAllExecutionsResponseMock()[0],
          id: "run-1",
          graph_id: agent.graph_id,
          status,
          started_at: new Date(Date.now() - 60_000),
          ended_at: status === AgentExecutionStatus.RUNNING ? null : new Date(),
        },
      ];
    }),
  );
}

function tile(label: string) {
  const button = screen.getByText(label).closest("button");
  expect(button).not.toBeNull();
  return within(button as HTMLElement);
}

// The server's side of the app's subscribes, noting each in `log`. It can't
// complete the first `failing.times` subscribes to `failing.graphID`, as when
// Redis doesn't confirm them in time; it confirms the rest. With `hold`, its
// answer to the next one waits for `release()`, and every answer after it.
function subscriptions(
  log: string[],
  failing?: { graphID: string; times: number; hold?: boolean },
) {
  const subscribed = new Set<string>();
  let failuresLeft = failing?.times ?? 0;
  let release = () => {};
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  return {
    subscribed,
    release: () => release(),
    async answer(message: SentMessage, socket: FakeWebSocket) {
      if (message.method !== "subscribe_graph_executions") return;
      const { graph_id } = message.data as { graph_id: string };
      if (graph_id === failing?.graphID && failuresLeft > 0) {
        failuresLeft -= 1;
        log.push(`failed ${graph_id}`);
        socket.receive({
          method: "error",
          success: false,
          error: "Could not handle 'subscribe_graph_executions'; retry",
          request: message,
          retryable: true,
        });
        return;
      }
      if (graph_id === failing?.graphID && failing.hold) await released;
      subscribed.add(graph_id);
      log.push(`subscribed ${graph_id}`);
      socket.receive({ method: message.method, success: true });
    },
  };
}

function noSpread() {
  // The first retry then waits 500 ms, and the reconnect spread is zero.
  const random = vi.spyOn(Math, "random").mockReturnValue(0);
  onTestFinished(() => random.mockRestore());
}

test("the live connection sends its sign-in token in the first message, never in its URL", async () => {
  serveLibrary(() => AgentExecutionStatus.RUNNING);
  const live = installLiveSocket();

  render(<LibraryPage />);
  const first = await live.acceptNext();
  await waitFor(() => expect(tile("Running now").getByText("1")).toBeDefined());
  await live.drop(first);
  const second = await live.acceptNext();
  await live.caughtUp();

  for (const socket of [first, second]) {
    expect(socket.url).not.toContain("?");
    expect(socket.url).not.toContain("test-token");
    expect(JSON.parse(socket.send.mock.calls[0][0])).toEqual({
      method: "authenticate",
      data: { token: "test-token" },
    });
  }
});

test("a run that finished while the live connection was down counts as finished once it reconnects", async () => {
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  serveLibrary(() => status);
  const live = installLiveSocket();

  render(<LibraryPage />);
  const first = await live.acceptNext();
  await waitFor(() => expect(tile("Running now").getByText("1")).toBeDefined());

  status = AgentExecutionStatus.COMPLETED;
  await live.drop(first);
  await live.acceptNext();

  await waitFor(
    () => expect(tile("Recently completed").getByText("1")).toBeDefined(),
    AFTER_RECONNECT,
  );
  expect(tile("Running now").getByText("0")).toBeDefined();
});

test("runs still loading when the live connection drops are read again once it reconnects", async () => {
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  const firstListHeld = new Promise<void>((resolve) => {
    releaseHeldList = resolve;
  });
  let requests = 0;
  serveLibrary(async () => {
    requests += 1;
    if (requests > 1) return status;
    // Read while the run was going, answered only after the reconnect.
    const read = status;
    await firstListHeld;
    return read;
  });
  const live = installLiveSocket();

  render(<LibraryPage />);
  const first = await live.acceptNext();
  // The summary's reconnect handler is in place once it has subscribed, and
  // its first read is out.
  await waitFor(() =>
    expect(
      first.send.mock.calls.some(
        ([data]) => JSON.parse(data).method === "subscribe_graph_executions",
      ),
    ).toBe(true),
  );
  await waitFor(() => expect(requests).toBe(1));

  status = AgentExecutionStatus.COMPLETED;
  await live.drop(first);
  await live.acceptNext();

  await waitFor(
    () => expect(tile("Recently completed").getByText("1")).toBeDefined(),
    AFTER_RECONNECT,
  );
  expect(tile("Running now").getByText("0")).toBeDefined();
});

test("a run update reaches Library once a subscribe the server could not complete is sent again", async () => {
  noSpread();
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  const log: string[] = [];
  serveLibrary(() => {
    log.push(`read ${status}`);
    return status;
  });
  const live = installLiveSocket();
  const runUpdates = subscriptions(log, { graphID: agent.graph_id, times: 1 });

  render(<LibraryPage />);
  const socket = await live.acceptNext(runUpdates.answer);
  await waitFor(
    () => expect(runUpdates.subscribed.has(agent.graph_id)).toBe(true),
    AFTER_RECONNECT,
  );
  // Confirmed late, so the page re-reads what it may have missed meanwhile.
  await live.caughtUp();
  await waitFor(() =>
    expect(log.slice(log.indexOf(`subscribed ${agent.graph_id}`))).toEqual([
      `subscribed ${agent.graph_id}`,
      "read RUNNING",
    ]),
  );

  status = AgentExecutionStatus.COMPLETED;
  await live.publish(socket, { id: "run-1", graph_id: agent.graph_id, status });
  await waitFor(() =>
    expect(tile("Recently completed").getByText("1")).toBeDefined(),
  );
});

test("after a reconnect, Library re-reads its runs at once, and again once a resubscription that failed is confirmed", async () => {
  noSpread();
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  const log: string[] = [];
  serveLibrary(() => {
    log.push(`read ${status}`);
    return status;
  });
  const live = installLiveSocket();

  render(<LibraryPage />);
  const first = await live.acceptNext(subscriptions(log).answer);
  await waitFor(() => expect(tile("Running now").getByText("1")).toBeDefined());

  status = AgentExecutionStatus.COMPLETED;
  await live.drop(first);
  const server = subscriptions(log, {
    graphID: agent.graph_id,
    times: 1,
    hold: true,
  });
  await live.acceptNext(server.answer);
  // Fresh right after the reconnect, while the resubscription has failed
  // and its retry waits for the server's answer.
  await waitFor(
    () => expect(tile("Recently completed").getByText("1")).toBeDefined(),
    AFTER_RECONNECT,
  );
  expect(server.subscribed.has(agent.graph_id)).toBe(false);

  log.length = 0;
  server.release();
  // Read again once the retry is confirmed, so the last read follows it:
  // read only before, the page would miss every later update of the run.
  await waitFor(() => {
    const confirmedAt = log.indexOf(`subscribed ${agent.graph_id}`);
    expect(confirmedAt).toBeGreaterThan(-1);
    expect(log.slice(confirmedAt + 1)).toContain("read COMPLETED");
  }, AFTER_RECONNECT);
});

test("the Library summary and the Activity list share one request for the runs after a reconnect", async () => {
  let status: AgentExecutionStatus = AgentExecutionStatus.RUNNING;
  serveLibrary(() => status);
  const live = installLiveSocket();

  render(
    <>
      <LibraryPage />
      <AccountMenuActivityRow />
    </>,
  );
  const first = await live.acceptNext();
  await waitFor(() => expect(tile("Running now").getByText("1")).toBeDefined());

  let requests = 0;
  function count({ request }: { request: Request }) {
    if (new URL(request.url).pathname.endsWith("/api/executions")) {
      requests += 1;
    }
  }
  server.events.on("request:start", count);
  try {
    status = AgentExecutionStatus.COMPLETED;
    await live.drop(first);
    await live.acceptNext();
    await waitFor(
      () => expect(tile("Recently completed").getByText("1")).toBeDefined(),
      AFTER_RECONNECT,
    );
    expect(requests).toBe(1);
  } finally {
    server.events.removeListener("request:start", count);
  }
});
