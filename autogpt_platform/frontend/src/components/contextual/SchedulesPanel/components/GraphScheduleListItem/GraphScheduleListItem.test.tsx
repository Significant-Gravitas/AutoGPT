import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { GraphScheduleListItem } from "./GraphScheduleListItem";

const schedule: GraphExecutionJobInfo = {
  id: "schedule",
  user_id: "owner",
  name: "Expert schedule",
  graph_id: "graph",
  graph_version: 1,
  cron: "0 9 * * *",
  input_data: {},
  next_run_time: "",
};

describe("GraphScheduleListItem", () => {
  it("shows a paused schedule with its name and actions", () => {
    render(<GraphScheduleListItem schedule={schedule} />);
    expect(screen.getByText("Expert schedule")).toBeDefined();
    expect(screen.getByText("Paused")).toBeDefined();
    expect(screen.queryByText("Pending")).toBeNull();
  });
});
