import {
  getGetV1GetExecutionDetailsQueryOptions,
  getGetV1ListAllExecutionsQueryKey,
  getV1GetExecutionDetails,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { getGetV1GetExecutionDetailsResponseMock } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { server } from "@/mocks/mock-server";
import {
  AFTER_RECONNECT,
  installLiveSocket,
} from "@/tests/integrations/live-socket";
import { act, screen, waitFor, within } from "@/tests/integrations/test-utils";
import { QueryClient } from "@tanstack/react-query";
import { render as renderWithoutProviders } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { AccountMenuActivityRow } from "../AccountMenuActivityRow";
import {
  OLD_RUN_STARTED,
  findRunOf,
  latestReadTaken,
  providersFor,
  run,
  serveActivity,
  showOldRunThenReconnect,
} from "./account-menu-activity-harness";

vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getWebSocketToken: async () => ({ token: "test-token" }),
}));

// Activity's reads by ID, so a test can wait until one has been taken.
vi.mock(
  "@/app/api/__generated__/endpoints/graphs/graphs",
  async (importOriginal) => {
    const actual =
      await importOriginal<
        typeof import("@/app/api/__generated__/endpoints/graphs/graphs")
      >();
    return {
      ...actual,
      getV1GetExecutionDetails: vi.fn(actual.getV1GetExecutionDetails),
    };
  },
);

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

beforeEach(() => {
  // latestReadTaken() must only ever see this test's reads.
  vi.mocked(getV1GetExecutionDetails).mockClear();
});

afterEach(() => {
  // Settle a held response even when the test failed before releasing it.
  releaseHeldList?.();
  releaseHeldList = undefined;
  vi.unstubAllGlobals();
});

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

test("a full list read without a reconnect reads no left-out run by ID", async () => {
  // Only a reconnect loses live updates, so only then is a run worth reading.
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const reads = vi.mocked(getV1GetExecutionDetails);
  await showOldRunThenReconnect(async () => AgentExecutionStatus.RUNNING, {
    queryClient,
    async beforeReconnect({ list }) {
      const readsBefore = reads.mock.calls.length;
      await act(() =>
        queryClient.refetchQueries({
          queryKey: getGetV1ListAllExecutionsQueryKey(),
        }),
      );
      // The list is in once its new run shows; a read would have gone out.
      expect(await within(list).findByText("New workflow")).toBeDefined();
      expect(reads.mock.calls.length).toBe(readsBefore);
    },
  });
});

test("a left-out run that is gone by the time it is read leaves Activity", async () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const { list } = await showOldRunThenReconnect(async () => 404, {
    queryClient,
  });

  await waitFor(() =>
    expect(within(list).queryByText("Old workflow")).toBeNull(),
  );

  // A later capped refetch that still leaves the run out must not bring
  // it back: the 404 read dropped its live-update entry from the log.
  await act(() =>
    queryClient.refetchQueries({
      queryKey: getGetV1ListAllExecutionsQueryKey(),
    }),
  );
  expect(await within(list).findByText("New workflow")).toBeDefined();
  await waitFor(() =>
    expect(within(list).queryByText("Old workflow")).toBeNull(),
  );
  await act(async () => {});
  expect(within(list).queryByText("Old workflow")).toBeNull();
});

test("a left-out run whose read fails stays as shown", async () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const reads = vi.mocked(getV1GetExecutionDetails);
  // Reads fail until the refetch below: the run stays as shown however
  // many reads the reconnect check takes.
  let answerFailedReads = true;
  const { list } = await showOldRunThenReconnect(
    async () => (answerFailedReads ? 500 : AgentExecutionStatus.COMPLETED),
    { queryClient },
  );
  await latestReadTaken();

  expect(
    (await findRunOf(list, "Old workflow")).getByText(/running$/),
  ).toBeDefined();

  // A later list refresh retries the failed read: the run is read again
  // and its answer shown.
  answerFailedReads = false;
  const readsBefore = reads.mock.calls.length;
  await act(() =>
    queryClient.refetchQueries({
      queryKey: getGetV1ListAllExecutionsQueryKey(),
    }),
  );
  await waitFor(() =>
    expect(reads.mock.calls.length).toBeGreaterThan(readsBefore),
  );
  await latestReadTaken();

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
  await latestReadTaken();

  expect(
    (await findRunOf(list, "Old workflow")).getByText(/^Stopped /),
  ).toBeDefined();
});

test("a read by ID doesn't overrule a list fetched after it started", async () => {
  let readStarted = false;
  let releaseRead = () => {};
  const readHeld = new Promise<void>((resolve) => {
    releaseRead = resolve;
  });
  releaseHeldList = releaseRead;
  const { list, live, second } = await showOldRunThenReconnect(
    async () => {
      readStarted = true;
      // Read while the run was going, answered after a newer list.
      await readHeld;
      return AgentExecutionStatus.RUNNING;
    },
    { listedAgainFrom: 3 },
  );
  await vi.waitFor(() => expect(readStarted).toBe(true));

  // The next list is fetched after the read started, and has the run done.
  await live.drop(second);
  await live.acceptNext();
  const oldRun = await findRunOf(list, "Old workflow");
  expect(
    await oldRun.findByText(/^Completed /, undefined, AFTER_RECONNECT),
  ).toBeDefined();
  releaseRead();
  await latestReadTaken();

  expect(
    (await findRunOf(list, "Old workflow")).getByText(/^Completed /),
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
  let poll: Promise<unknown> = Promise.resolve();
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
        poll = queryClient
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
  // Activity reads the run with a request of its own while the poll waits.
  await vi.waitFor(() => expect(reads).toBe(2));
  await latestReadTaken();
  releasePoll();
  // Whatever took the poll's answer has handled it once the poll settles.
  await poll;
  await act(async () => {});

  const oldRun = await findRunOf(list, "Old workflow");
  expect(oldRun.getByText(/^Awaiting approval /)).toBeDefined();
  expect(oldRun.queryByText(/running$/)).toBeNull();
});

test("a run that finished while the menu was closed is not shown as running on reopen", async () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const reads = vi.mocked(getV1GetExecutionDetails);
  const newer = Array.from({ length: 250 }, (_, i) =>
    run(`run-${i}`, AgentExecutionStatus.COMPLETED, "graph-busy"),
  );
  // The list never includes the old run: only the shared log could show it.
  serveActivity(async () => newer, {
    "graph-old": "Old workflow",
    "graph-busy": "Busy workflow",
  });
  server.use(
    http.get(
      "/api/proxy/api/graphs/:graphId/executions/:graphExecId",
      async () =>
        HttpResponse.json({
          ...getGetV1GetExecutionDetailsResponseMock(),
          ...run("run-old", AgentExecutionStatus.RUNNING, "graph-old"),
          inputs: {},
          outputs: {},
          node_executions: [],
        }),
    ),
  );
  const live = installLiveSocket();

  const firstRender = renderWithoutProviders(<AccountMenuActivityRow />, {
    wrapper: providersFor(queryClient),
  });
  await userEvent.click(
    await screen.findByTestId("account-menu-activity-trigger"),
  );
  const list = await screen.findByTestId("agent-activity-dropdown");
  const first = await live.acceptNext();
  await live.publish(first, {
    ...run("run-old", AgentExecutionStatus.RUNNING, "graph-old"),
    started_at: OLD_RUN_STARTED,
  });
  expect(
    (await findRunOf(list, "Old workflow")).getByText(/running$/),
  ).toBeDefined();

  // The menu closes while the run is still shown as going. The unmount
  // resets the shared log, so the stale entry is gone on reopen.
  firstRender.unmount();
  renderWithoutProviders(<AccountMenuActivityRow />, {
    wrapper: providersFor(queryClient),
  });
  await userEvent.click(
    await screen.findByTestId("account-menu-activity-trigger"),
  );
  const reopened = await screen.findByTestId("agent-activity-dropdown");
  // The reopened list has landed once a listed run shows.
  await waitFor(() =>
    expect(
      within(reopened).queryAllByText("Busy workflow").length,
    ).toBeGreaterThan(0),
  );
  await act(async () => {});
  expect(within(reopened).queryByText("Old workflow")).toBeNull();
  expect(reads.mock.calls.length).toBe(0);
});

test("a read that settles after the menu closed does not bring its run back", async () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const reads = vi.mocked(getV1GetExecutionDetails);
  let readStarted = false;
  let releaseRead = () => {};
  const readHeld = new Promise<void>((resolve) => {
    releaseRead = resolve;
  });
  releaseHeldList = releaseRead;
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
    http.get(
      "/api/proxy/api/graphs/:graphId/executions/:graphExecId",
      async () => {
        readStarted = true;
        await readHeld;
        return HttpResponse.json({
          ...getGetV1GetExecutionDetailsResponseMock(),
          ...run("run-old", AgentExecutionStatus.RUNNING, "graph-old"),
          inputs: {},
          outputs: {},
          node_executions: [],
        });
      },
    ),
  );
  const live = installLiveSocket();

  const firstRender = renderWithoutProviders(<AccountMenuActivityRow />, {
    wrapper: providersFor(queryClient),
  });
  await userEvent.click(
    await screen.findByTestId("account-menu-activity-trigger"),
  );
  const list = await screen.findByTestId("agent-activity-dropdown");
  const first = await live.acceptNext();
  await live.publish(first, {
    ...run("run-old", AgentExecutionStatus.RUNNING, "graph-old"),
    started_at: OLD_RUN_STARTED,
  });
  expect(
    (await findRunOf(list, "Old workflow")).getByText(/running$/),
  ).toBeDefined();

  // The reconnect starts a read of the left-out run; hold it, then close
  // the menu while it is out.
  await live.drop(first);
  await live.acceptNext();
  expect(await findRunOf(list, "New workflow", AFTER_RECONNECT)).toBeDefined();
  await vi.waitFor(() => expect(readStarted).toBe(true));

  firstRender.unmount();
  // The read settles after the unmount reset: its answer must not land in
  // the fresh log or revive the run.
  releaseRead();
  await latestReadTaken();

  renderWithoutProviders(<AccountMenuActivityRow />, {
    wrapper: providersFor(queryClient),
  });
  await userEvent.click(
    await screen.findByTestId("account-menu-activity-trigger"),
  );
  const reopened = await screen.findByTestId("agent-activity-dropdown");
  expect(await findRunOf(reopened, "New workflow")).toBeDefined();
  await act(async () => {});
  expect(within(reopened).queryByText("Old workflow")).toBeNull();
  // No fresh read went out on reopen: the stale answer satisfied nothing.
  expect(reads.mock.calls.length).toBe(1);
});
