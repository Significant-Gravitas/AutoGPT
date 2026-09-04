import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";

const pathnameMock = vi.hoisted(() => ({ current: "/marketplace" }));

vi.mock("next/navigation", () => ({
  usePathname: () => pathnameMock.current,
}));

afterEach(() => {
  pathnameMock.current = "/marketplace";
});

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
  Confetti: () => <div data-testid="confetti" />,
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

  it("hides the redundant 'Go to Creator Dashboard' button on the dashboard", () => {
    pathnameMock.current = "/settings/creator-dashboard";
    const onEdit = vi.fn();
    render(
      <AgentReviewStep
        {...baseProps}
        status={SubmissionStatus.PENDING}
        onEdit={onEdit}
      />,
    );

    // On the dashboard the button would just navigate back to the same page,
    // so it is hidden; editing and "Done" remain available.
    expect(screen.queryByTestId("view-progress-button")).toBeNull();
    expect(screen.getByTestId("edit-submission-button")).toBeDefined();
    expect(screen.getByText("Done")).toBeDefined();
  });

  it("celebrates a fresh pending submission but not one viewed on the dashboard", () => {
    const { unmount } = render(
      <AgentReviewStep {...baseProps} status={SubmissionStatus.PENDING} />,
    );
    // Post-publish (off-dashboard) pending keeps the celebration.
    expect(screen.getByTestId("confetti")).toBeDefined();
    unmount();

    pathnameMock.current = "/settings/creator-dashboard";
    render(
      <AgentReviewStep {...baseProps} status={SubmissionStatus.PENDING} />,
    );
    // Viewing an existing pending submission should not throw confetti.
    expect(screen.queryByTestId("confetti")).toBeNull();
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

  it("renders the draft hero without an edit action", () => {
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

    // Editing is only supported for pending submissions, and "Done" is the only
    // footer action.
    expect(screen.queryByTestId("edit-submission-button")).toBeNull();
    expect(screen.queryByTestId("view-progress-button")).toBeNull();
    expect(screen.getByText("Done")).toBeDefined();
  });

  it("renders the rejected hero + feedback without an edit action", () => {
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

    // Editing is only supported for pending submissions, and "Done" is the only
    // footer action.
    expect(screen.queryByTestId("edit-submission-button")).toBeNull();
    expect(screen.queryByTestId("view-progress-button")).toBeNull();
    expect(screen.getByText("Done")).toBeDefined();
  });
});
