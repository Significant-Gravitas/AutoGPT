import {
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1GetExecutionDetailsResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { server } from "@/mocks/mock-server";
import { expect, test, vi } from "vitest";
import { createOmittedRunChecker } from "../checkOmittedRuns";

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
  const log = { startRead: vi.fn(), finishRead: vi.fn(() => true) };
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
