import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";

vi.mock("next/navigation", () => ({
  usePathname: () => "/marketplace",
}));

vi.mock("@/components/molecules/Confetti/Confetti", () => ({
  Confetti: () => null,
}));

import { AgentReviewStep } from "../AgentReviewStep";

const baseProps = {
  agentName: "Test Agent",
  subheader: "A subheader",
  description: "A description",
  onClose: vi.fn(),
  onDone: vi.fn(),
  onViewProgress: vi.fn(),
};

describe("AgentReviewStep", () => {
  it("renders the pending hero, the timeline, and the footer", () => {
    const onDone = vi.fn();
    const onViewProgress = vi.fn();
    render(
      <AgentReviewStep
        {...baseProps}
        onDone={onDone}
        onViewProgress={onViewProgress}
      />,
    );

    expect(screen.getByText("Submission received")).toBeDefined();
    expect(screen.getByText("Test Agent")).toBeDefined();

    // Timeline copy from the SECRT-2324 refresh.
    expect(screen.getByText("Submitted for review")).toBeDefined();
    expect(screen.getByText("Reviewed soon")).toBeDefined();
    expect(screen.getByText("Approved listings go live")).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(onDone).toHaveBeenCalled();

    fireEvent.click(screen.getByTestId("view-progress-button"));
    expect(onViewProgress).toHaveBeenCalled();
  });

  it("renders the approved hero when status is APPROVED", () => {
    render(
      <AgentReviewStep {...baseProps} status={SubmissionStatus.APPROVED} />,
    );
    expect(screen.getByText("Agent approved")).toBeDefined();
  });

  it("renders the rejected hero + review comments and hides the timeline", () => {
    render(
      <AgentReviewStep
        {...baseProps}
        status={SubmissionStatus.REJECTED}
        reviewComments="Please clarify your description."
      />,
    );
    expect(screen.getByText("Agent needs changes")).toBeDefined();
    expect(screen.getByText("Please clarify your description.")).toBeDefined();
    expect(screen.queryByText("What happens next")).toBeNull();
  });
});
