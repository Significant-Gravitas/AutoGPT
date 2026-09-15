import { expect, test, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { getGetV2ListMySubmissionsResponseMock } from "@/app/api/__generated__/endpoints/store/store.msw";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { AgentTableRow } from "../AgentTableRow";

vi.mock("next/image", async () =>
  vi.importActual<typeof import("next/image")>("next/image"),
);

const LOCAL_URL = "/api/store/media/agent-1/images/cover.png";

test("a locally served image renders unoptimized and still falls back when it fails", () => {
  const submission = {
    ...getGetV2ListMySubmissionsResponseMock().submissions[0],
    graph_id: "graph-1",
    graph_version: 7,
    listing_version_id: "listing-approved",
    name: "Local media agent",
    status: SubmissionStatus.APPROVED,
    image_urls: [LOCAL_URL],
    submitted_at: new Date("2026-01-01T00:00:00.000Z"),
  };

  const { container } = render(
    <AgentTableRow
      storeAgentSubmission={submission}
      onViewSubmission={vi.fn()}
      onDeleteSubmission={vi.fn()}
      onEditSubmission={vi.fn()}
    />,
  );

  const image = screen.getByAltText(submission.name);
  expect(image.getAttribute("src")).toContain(LOCAL_URL);
  expect(image.getAttribute("src")).not.toContain("/_next/image");
  expect(image.hasAttribute("srcset")).toBe(false);

  fireEvent.error(image);

  expect(screen.queryByAltText(submission.name)).toBeNull();
  expect(container.querySelector("svg")).not.toBeNull();
});
