import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";

vi.mock("next/navigation", () => ({
  usePathname: () => "/marketplace",
}));

vi.mock("next/link", () => ({
  default: ({
    href,
    children,
    ...rest
  }: {
    href: string;
    children: ReactNode;
  }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  ),
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
  it("renders the pending hero, the review stepper, and the footer", () => {
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

    // Stepper reflects the current step.
    expect(screen.getByText("What happens next")).toBeDefined();
    expect(screen.getByText("Submitted for review")).toBeDefined();
    expect(screen.getByText("Goes live")).toBeDefined();
    expect(
      screen.getByText("Typically reviewed within 2–3 days."),
    ).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: "Done" }));
    expect(onDone).toHaveBeenCalled();

    fireEvent.click(screen.getByTestId("view-progress-button"));
    expect(onViewProgress).toHaveBeenCalled();
  });

  it("renders submission metadata (version, category, submitted date)", () => {
    render(
      <AgentReviewStep
        {...baseProps}
        status={SubmissionStatus.PENDING}
        version={3}
        category="Productivity"
        submittedAt="2026-07-01T10:00:00Z"
      />,
    );

    expect(screen.getByTestId("submission-meta")).toBeDefined();
    expect(screen.getByText("Version")).toBeDefined();
    expect(screen.getByText("v3")).toBeDefined();
    expect(screen.getByText("Category")).toBeDefined();
    expect(screen.getByText("Productivity")).toBeDefined();
    expect(screen.getByText("Submitted")).toBeDefined();
  });

  it("shows an 'Edit details' action for pending submissions", () => {
    const onEdit = vi.fn();
    render(
      <AgentReviewStep
        {...baseProps}
        status={SubmissionStatus.PENDING}
        onEdit={onEdit}
      />,
    );
    const edit = screen.getByTestId("edit-submission-button");
    expect(edit.textContent).toContain("Edit details");
    fireEvent.click(edit);
    expect(onEdit).toHaveBeenCalled();
    // View progress is still the primary action for pending.
    expect(screen.getByTestId("view-progress-button")).toBeDefined();
  });

  it("renders the approved hero, hides the stepper, and shows runs + share link", () => {
    render(
      <AgentReviewStep
        {...baseProps}
        status={SubmissionStatus.APPROVED}
        reviewedAt="2026-07-02T10:00:00Z"
        runCount={1234}
        marketplaceUrl="/marketplace/agent/creator/test-agent"
      />,
    );
    expect(screen.getByText("Agent approved")).toBeDefined();
    expect(screen.queryByText("What happens next")).toBeNull();
    expect(screen.getByText("Live since")).toBeDefined();
    expect(screen.getByText("Runs")).toBeDefined();
    expect(screen.getByText("1,234")).toBeDefined();
    expect(screen.getByTestId("copy-share-link-button")).toBeDefined();
  });

  it("shows a marketplace CTA when the submission is approved and live", () => {
    render(
      <AgentReviewStep
        {...baseProps}
        status={SubmissionStatus.APPROVED}
        marketplaceUrl="/marketplace/agent/creator/test-agent"
      />,
    );
    const cta = screen.getByTestId("view-marketplace-button");
    expect(cta.getAttribute("href")).toBe(
      "/marketplace/agent/creator/test-agent",
    );
    expect(screen.queryByTestId("view-progress-button")).toBeNull();
  });

  it("renders the draft hero with a 'Continue editing' primary CTA", () => {
    const onEdit = vi.fn();
    render(
      <AgentReviewStep
        {...baseProps}
        status={SubmissionStatus.DRAFT}
        onEdit={onEdit}
      />,
    );
    expect(screen.getByText("Draft saved")).toBeDefined();
    expect(screen.queryByText("What happens next")).toBeNull();
    expect(screen.getByText("Not submitted yet")).toBeDefined();

    const cta = screen.getByTestId("edit-submission-button");
    expect(cta.textContent).toContain("Continue editing");
    fireEvent.click(cta);
    expect(onEdit).toHaveBeenCalled();
    expect(screen.queryByTestId("view-progress-button")).toBeNull();
  });

  it("renders the rejected hero + feedback with an 'Edit & resubmit' CTA", () => {
    const onEdit = vi.fn();
    render(
      <AgentReviewStep
        {...baseProps}
        status={SubmissionStatus.REJECTED}
        reviewComments="Please clarify your description."
        reviewedAt="2026-07-03T10:00:00Z"
        onEdit={onEdit}
      />,
    );
    expect(screen.getByText("Agent needs changes")).toBeDefined();
    expect(screen.getByText("Please clarify your description.")).toBeDefined();
    expect(screen.queryByText("What happens next")).toBeNull();
    expect(screen.getByText("Reviewed")).toBeDefined();

    const cta = screen.getByTestId("edit-submission-button");
    expect(cta.textContent).toContain("Edit & resubmit");
    fireEvent.click(cta);
    expect(onEdit).toHaveBeenCalled();
  });
});
