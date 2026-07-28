import { describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";

import { getGetV1ListExecutionSchedulesForAGraphMockHandler } from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { SelectedScheduleView } from "./SelectedScheduleView";

const agent = {
  id: "lib-agent-1",
  name: "Nightly cleanup",
  graph_id: "graph-abc",
  graph_version: 1,
} as unknown as LibraryAgent;

function makeSchedule(
  overrides: Partial<GraphExecutionJobInfo> = {},
): GraphExecutionJobInfo {
  return {
    id: "sched-1",
    name: "Daily summary",
    user_id: "user-1",
    graph_id: "graph-abc",
    graph_version: 1,
    agent_name: "Daily summary agent",
    cron: "0 9 * * *",
    input_data: {},
    next_run_time: new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString(),
    timezone: "UTC",
    ...overrides,
  };
}

describe("SelectedScheduleView", () => {
  test("payment-lapsed schedule shows the paused label instead of the next-run time", async () => {
    server.use(
      getGetV1ListExecutionSchedulesForAGraphMockHandler([
        makeSchedule({
          is_paused: true,
          next_run_time: "",
          paused_reason: "payment_lapsed",
        }),
      ]),
    );

    render(<SelectedScheduleView agent={agent} scheduleId="sched-1" />);

    expect(await screen.findByText("Paused — payment required")).toBeDefined();
  });

  test("schedule with no next run time shows Pending", async () => {
    server.use(
      getGetV1ListExecutionSchedulesForAGraphMockHandler([
        makeSchedule({ is_paused: false, next_run_time: "" }),
      ]),
    );

    render(<SelectedScheduleView agent={agent} scheduleId="sched-1" />);

    expect(await screen.findByText("Pending")).toBeDefined();
    expect(screen.queryByText("Paused — payment required")).toBeNull();
  });
});
