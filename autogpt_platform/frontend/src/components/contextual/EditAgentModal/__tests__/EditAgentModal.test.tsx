import { describe, expect, it, vi } from "vitest";

import { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { EditAgentModal } from "../EditAgentModal";

function makeSubmission(): StoreSubmissionEditRequest & {
  store_listing_version_id: string | undefined;
  graph_id: string;
} {
  return {
    name: "Edit Me Agent",
    sub_heading: "A subhead",
    description: "Some description that explains the agent",
    image_urls: ["https://cdn.test/a.png"],
    video_url: "https://www.youtube.com/watch?v=abc123",
    agent_output_demo_url: "https://www.youtube.com/watch?v=demo123",
    categories: ["productivity"],
    changes_summary: "Initial edit summary",
    store_listing_version_id: "lv-1",
    graph_id: "graph-1",
  };
}

describe("EditAgentModal", () => {
  it("does not render when submission is null", () => {
    render(
      <EditAgentModal
        isOpen={true}
        onClose={() => {}}
        submission={null}
        onSuccess={() => {}}
      />,
    );
    expect(screen.queryByTestId("edit-agent-modal")).toBeNull();
  });

  it("renders the new accordion sections with the update note when open", () => {
    render(
      <EditAgentModal
        isOpen={true}
        onClose={() => {}}
        submission={makeSubmission()}
        onSuccess={() => {}}
      />,
    );

    expect(screen.getByTestId("edit-agent-modal")).toBeDefined();
    expect(screen.getByText(/update note/i)).toBeDefined();
    expect(screen.getByText(/listing basics/i)).toBeDefined();
    expect(screen.getByText(/thumbnails/i)).toBeDefined();
    expect(screen.getByText(/experience details/i)).toBeDefined();
    expect(
      screen.getByRole("button", { name: /update submission/i }),
    ).toBeDefined();
  });

  it("invokes onClose when the Cancel button is clicked", () => {
    const onClose = vi.fn();
    render(
      <EditAgentModal
        isOpen={true}
        onClose={onClose}
        submission={makeSubmission()}
        onSuccess={() => {}}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));
    expect(onClose).toHaveBeenCalled();
  });
});
