import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("./useSelectedScheduleActions", () => ({
  useSelectedScheduleActions: () => ({
    openInBuilderHref: null,
    showDeleteDialog: false,
    setShowDeleteDialog: vi.fn(),
    handleDelete: vi.fn(),
    isDeleting: false,
    handleRunNow: vi.fn(),
    isRunning: false,
  }),
}));

vi.mock("../../../AgentActionsDropdown", () => ({
  AgentActionsDropdown: () => <div />,
}));

import { SelectedScheduleActions } from "./SelectedScheduleActions";

const agent = { graph_id: "graph-1" } as LibraryAgent;
const schedule = {
  id: "schedule-1",
  graph_id: "graph-1",
  graph_version: 1,
} as GraphExecutionJobInfo;

function renderActions(canDelete: boolean) {
  return render(
    <TooltipProvider>
      <SelectedScheduleActions
        agent={agent}
        scheduleId={schedule.id}
        schedule={schedule}
        canDelete={canDelete}
      />
    </TooltipProvider>,
  );
}

describe("SelectedScheduleActions", () => {
  it("hides delete when the viewer does not own the schedule", () => {
    renderActions(false);

    expect(
      screen.queryByRole("button", { name: "Delete schedule" }),
    ).toBeNull();
  });

  it("shows delete for the schedule owner", () => {
    renderActions(true);

    expect(screen.getByRole("button", { name: "Delete schedule" })).not.toBe(
      null,
    );
  });
});
