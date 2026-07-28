import { describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";

import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { ScheduleListItem } from "./ScheduleListItem";

const agent = {
  id: "lib-agent-1",
  graph_id: "graph-1",
  graph_version: 1,
} as unknown as LibraryAgent;

function makeSchedule(
  overrides: Partial<GraphExecutionJobInfo> = {},
): GraphExecutionJobInfo {
  return {
    id: "sched-1",
    user_id: "user-1",
    graph_id: "graph-1",
    graph_version: 1,
    name: "Daily run",
    cron: "0 0 * * *",
    input_data: {},
    next_run_time: "2099-01-01T00:00:00.000Z",
    ...overrides,
  };
}

describe("ScheduleListItem", () => {
  test("payment-lapsed schedule shows the paused description instead of a run time", async () => {
    render(
      <ScheduleListItem
        schedule={makeSchedule({
          is_paused: true,
          next_run_time: "",
          paused_reason: "payment_lapsed",
        })}
        agent={agent}
      />,
    );

    expect(await screen.findByText("Paused — payment required")).toBeDefined();
  });

  test("schedule without a next run time falls back to Pending", async () => {
    render(
      <ScheduleListItem
        schedule={makeSchedule({ is_paused: false, next_run_time: "" })}
        agent={agent}
      />,
    );

    expect(await screen.findByText("Pending")).toBeDefined();
    expect(screen.queryByText("Paused — payment required")).toBeNull();
  });

  test("active schedule shows a relative next-run time", async () => {
    render(<ScheduleListItem schedule={makeSchedule()} agent={agent} />);

    expect(await screen.findByText("Daily run")).toBeDefined();
    expect(screen.queryByText("Paused — payment required")).toBeNull();
    expect(screen.queryByText("Pending")).toBeNull();
  });
});
