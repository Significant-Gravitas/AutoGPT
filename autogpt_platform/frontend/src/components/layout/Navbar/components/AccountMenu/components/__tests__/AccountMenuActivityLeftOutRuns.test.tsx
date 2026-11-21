import {
  getGetV1GetExecutionDetailsQueryOptions,
  getGetV1ListAllExecutionsQueryKey,
  getV1GetExecutionDetails,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { AFTER_RECONNECT } from "@/tests/integrations/live-socket";
import { act, waitFor, within } from "@/tests/integrations/test-utils";
import { QueryClient } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import {
  OLD_RUN_STARTED,
  findRunOf,
  latestReadTaken,
  run,
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
  const { list } = await showOldRunThenReconnect(async () => 500);
  await latestReadTaken();

  expect(
    (await findRunOf(list, "Old workflow")).getByText(/running$/),
  ).toBeDefined();
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
