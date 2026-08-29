import { render, screen, fireEvent } from "@testing-library/react";
import { getGetV2ListMySubmissionsResponseMock } from "@/app/api/__generated__/endpoints/store/store.msw";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { AgentTableRow } from "../AgentTableRow";
import { beforeEach, describe, expect, test, vi } from "vitest";

function makeSubmission(status: SubmissionStatus) {
  const submission = getGetV2ListMySubmissionsResponseMock().submissions[0];

  return {
    ...submission,
    graph_id: "graph-1",
    graph_version: 7,
    listing_version_id: `listing-${status.toLowerCase()}`,
    name: `Agent ${status}`,
    description: `Description ${status}`,
    status,
    image_urls: [],
    submitted_at: new Date("2026-01-01T00:00:00.000Z"),
  };
}

describe("AgentTableRow", () => {
  const onViewSubmission = vi.fn();
  const onDeleteSubmission = vi.fn();
  const onEditSubmission = vi.fn();

  beforeEach(() => {
    onViewSubmission.mockReset();
    onDeleteSubmission.mockReset();
    onEditSubmission.mockReset();
  });

  test("shows edit and delete actions for pending submissions", async () => {
    render(
      <AgentTableRow
        storeAgentSubmission={makeSubmission(SubmissionStatus.PENDING)}
        onViewSubmission={onViewSubmission}
        onDeleteSubmission={onDeleteSubmission}
        onEditSubmission={onEditSubmission}
      />,
    );

    fireEvent.pointerDown(screen.getByTestId("agent-table-row-actions"));

    fireEvent.click(await screen.findByText("Edit"));
    expect(onEditSubmission).toHaveBeenCalledTimes(1);

    fireEvent.pointerDown(screen.getByTestId("agent-table-row-actions"));
    fireEvent.click(await screen.findByText("Delete"));
    expect(onDeleteSubmission).toHaveBeenCalledWith("listing-pending");
    expect(onViewSubmission).not.toHaveBeenCalled();
  });

  test("shows view only for non-pending submissions", async () => {
    const approvedSubmission = makeSubmission(SubmissionStatus.APPROVED);

    render(
      <AgentTableRow
        storeAgentSubmission={approvedSubmission}
        onViewSubmission={onViewSubmission}
        onDeleteSubmission={onDeleteSubmission}
        onEditSubmission={onEditSubmission}
      />,
    );

    fireEvent.pointerDown(screen.getByTestId("agent-table-row-actions"));

    const viewAction = await screen.findByText("View");
    fireEvent.click(viewAction);

    expect(onViewSubmission).toHaveBeenCalledWith(approvedSubmission);
    expect(screen.queryByText("Edit")).toBeNull();
    expect(screen.queryByText("Delete")).toBeNull();
  });
});
