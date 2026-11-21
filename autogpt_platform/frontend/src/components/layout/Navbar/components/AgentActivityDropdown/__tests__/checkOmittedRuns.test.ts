import {
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1GetExecutionDetailsResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import { createOmittedRunChecker } from "../checkOmittedRuns";
import { createLiveUpdateLog } from "../useLiveUpdateLog";

function running(id: string): GraphExecutionMeta {
  return {
    ...getGetV1GetExecutionDetailsResponseMock(),
    id,
    graph_id: `graph-${id}`,
    status: AgentExecutionStatus.RUNNING,
  };
}

test("each full list reads the left-out runs read longest ago first", async () => {
  const runs = Array.from({ length: 11 }, (_, i) => running(`old-${i}`));
  const requested: string[] = [];
  server.use(
    getGetV1GetExecutionDetailsMockHandler(({ params }) => {
      requested.push(String(params.graphExecId));
      return {
        ...running(String(params.graphExecId)),
        inputs: {},
        outputs: {},
      };
    }),
  );
  const checker = createOmittedRunChecker();
  const log = {
    startRead: vi.fn(),
    finishRead: vi.fn(() => ({ accepted: true })),
    currentSequence: vi.fn(() => 0),
  };
  const check = () =>
    checker.check({ runs, log, onRead: vi.fn(), onGone: vi.fn() });

  await check();
  expect(requested).toHaveLength(10);
  expect(requested).not.toContain("old-10");

  requested.length = 0;
  await check();
  expect(requested).toHaveLength(10);
  expect(requested).toContain("old-10");
});

test("a second check while a read is out queues a fresh read for it", async () => {
  const runs = [running("run-1")];
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  let requests = 0;
  server.use(
    getGetV1GetExecutionDetailsMockHandler(async ({ params }) => {
      requests += 1;
      if (requests === 1) await held;
      return {
        ...running(String(params.graphExecId)),
        inputs: {},
        outputs: {},
      };
    }),
  );
  const checker = createOmittedRunChecker();
  const log = {
    startRead: vi.fn(),
    finishRead: vi.fn(() => ({ accepted: true })),
    currentSequence: vi.fn(() => 0),
  };
  const onRead = vi.fn();
  const check = () => checker.check({ runs, log, onRead, onGone: vi.fn() });

  const first = check();
  try {
    await vi.waitFor(() => expect(requests).toBe(1));
    // A second reconnect while the first read is still out: the run's answer
    // predates it, so a fresh read is queued for when the first settles.
    const second = check();
    await second;
  } finally {
    // A failure above must not leave the held response pending: the MSW
    // resolver stays blocked and request-draining teardown pays for it.
    release();
    await first;
  }

  await vi.waitFor(() => expect(requests).toBe(2));
  expect(onRead).toHaveBeenCalledTimes(2);
});

test("a queued fresh read is dropped when a newer list settles the run first", async () => {
  const runs = [running("run-1")];
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  let requests = 0;
  server.use(
    getGetV1GetExecutionDetailsMockHandler(async ({ params }) => {
      requests += 1;
      if (requests === 1) await held;
      return {
        ...running(String(params.graphExecId)),
        inputs: {},
        outputs: {},
      };
    }),
  );
  const checker = createOmittedRunChecker();
  // The first read's answer is refused: a newer list settled the run first.
  const log = {
    startRead: vi.fn(),
    // The newer list arrived after the queued check was asked.
    finishRead: vi.fn(() => ({ accepted: false, supersededAt: 1 })),
    currentSequence: vi.fn(() => 0),
  };
  const onRead = vi.fn();
  const check = () => checker.check({ runs, log, onRead, onGone: vi.fn() });

  const first = check();
  try {
    await vi.waitFor(() => expect(requests).toBe(1));
    const second = check();
    await second;
  } finally {
    // A failure above must not leave the held response pending: the MSW
    // resolver stays blocked and request-draining teardown pays for it.
    release();
    await first;
  }

  expect(requests).toBe(1);
  expect(onRead).not.toHaveBeenCalled();
});

test("a queued fresh read survives when the superseding update predates it", async () => {
  const runs = [running("run-1")];
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  let requests = 0;
  server.use(
    getGetV1GetExecutionDetailsMockHandler(async ({ params }) => {
      requests += 1;
      // The first read is held; the queued fresh read answers at once,
      // showing the run finished while the tab was disconnected.
      if (requests === 1) await held;
      return {
        ...running(String(params.graphExecId)),
        status:
          requests === 1
            ? AgentExecutionStatus.RUNNING
            : AgentExecutionStatus.COMPLETED,
        inputs: {},
        outputs: {},
      };
    }),
  );
  const checker = createOmittedRunChecker();
  // The real log, so the ordering is the log's own: a live update arrives
  // during the first read, then a second reconnect queues a fresh read.
  const log = createLiveUpdateLog();
  const onRead = vi.fn();
  const check = () => checker.check({ runs, log, onRead, onGone: vi.fn() });

  const first = check();
  try {
    await vi.waitFor(() => expect(requests).toBe(1));
    // A live update arrives during the first read: its answer is refused.
    log.recordUpdate({
      id: "run-1",
      status: AgentExecutionStatus.RUNNING,
    } as GraphExecution);
    // A second reconnect while the first read is still out: the run's answer
    // predates it, so a fresh read is queued — and the earlier live update
    // does not drop it, since it predates the queued check.
    const second = check();
    await second;
  } finally {
    // A failure above must not leave the held response pending: the MSW
    // resolver stays blocked and request-draining teardown pays for it.
    release();
    await first;
  }

  await vi.waitFor(() => expect(requests).toBe(2));
  // Only the fresh read's answer is applied: the run finished while the
  // tab was disconnected, so it must not stay shown as running.
  expect(onRead).toHaveBeenCalledTimes(1);
  expect(onRead).toHaveBeenCalledWith(
    expect.objectContaining({
      id: "run-1",
      status: AgentExecutionStatus.COMPLETED,
    }),
  );
});

test("a later list refresh reads the runs the reconnect check never got to", async () => {
  const runs = Array.from({ length: 11 }, (_, i) => running(`old-${i}`));
  const requested: string[] = [];
  server.use(
    getGetV1GetExecutionDetailsMockHandler(({ params }) => {
      requested.push(String(params.graphExecId));
      return {
        ...running(String(params.graphExecId)),
        inputs: {},
        outputs: {},
      };
    }),
  );
  const checker = createOmittedRunChecker();
  const log = {
    startRead: vi.fn(),
    finishRead: vi.fn(() => ({ accepted: true })),
    currentSequence: vi.fn(() => 0),
  };
  const args = { runs, log, onRead: vi.fn(), onGone: vi.fn() };

  await checker.check(args);
  expect(requested).toHaveLength(10);
  expect(requested).not.toContain("old-10");

  await checker.followUp(args);
  expect(requested).toContain("old-10");
  expect(requested).toHaveLength(11);

  // A second follow-up reads nothing more: every run has been read.
  await checker.followUp(args);
  expect(requested).toHaveLength(11);
});

test("a later list refresh retries a failed read, a bounded number of times", async () => {
  const runs = [running("run-1")];
  let requests = 0;
  // No generated 500 handler exists for this endpoint, so the error comes
  // from a raw handler on the same path.
  server.use(
    http.get("/api/proxy/api/graphs/:graphId/executions/:graphExecId", () => {
      requests += 1;
      return HttpResponse.json({ detail: "boom" }, { status: 500 });
    }),
  );
  const checker = createOmittedRunChecker();
  const log = {
    startRead: vi.fn(),
    finishRead: vi.fn(() => ({ accepted: true })),
    currentSequence: vi.fn(() => 0),
  };
  const args = { runs, log, onRead: vi.fn(), onGone: vi.fn() };

  await checker.check(args);
  expect(requests).toBe(1);
  await checker.followUp(args);
  expect(requests).toBe(2);
  await checker.followUp(args);
  expect(requests).toBe(3);
  // Three failed attempts: the next follow-up stops retrying.
  await checker.followUp(args);
  expect(requests).toBe(3);
});
