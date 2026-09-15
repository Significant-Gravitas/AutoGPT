import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
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

  it("says Paused in the View dialog too, not Pending", async () => {
    // The row and the dialog used to derive this separately, so a paused
    // schedule read "Paused" in the row and "Next run: Pending" on opening it.
    render(<GraphScheduleListItem schedule={schedule} />);
    await userEvent.setup().click(screen.getByLabelText("View schedule"));

    expect(await screen.findByText("Next run")).toBeDefined();
    expect(screen.queryByText("Pending")).toBeNull();
    expect(screen.getAllByText("Paused").length).toBeGreaterThan(1);
  });
});
