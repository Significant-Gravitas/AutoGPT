import { describe, expect, test, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";
import { TaskGroups } from "../WalletTaskGroups";
import type { TaskGroup } from "../../Wallet";

const mockOnboardingState = {
  completedSteps: ["COMPLETED_TASK"] as string[],
  notified: [] as string[],
  walletShown: true,
  userId: "test-user",
  rewardedFor: [],
  usageReason: null,
  integrations: [],
  otherIntegrations: null,
  selectedStoreListingVersionId: null,
  agentInput: null,
  onboardingAgentExecutionId: null,
  agentRuns: 0,
  lastRunAt: null,
  consecutiveRunDays: 0,
};

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  useOnboarding: () => ({
    state: mockOnboardingState,
    updateState: vi.fn(),
  }),
}));

vi.mock("canvas-confetti", () => ({
  __esModule: true,
  default: vi.fn(),
}));

const groups: TaskGroup[] = [
  {
    name: "Getting Started",
    details: "Get to your first successful agent run.",
    tasks: [
      {
        id: "VISIT_COPILOT",
        name: "Visit the Copilot",
        amount: 5,
        details: "Open the Copilot to start your AI automation journey",
      },
      {
        id: "SHARE_PLATFORM",
        name: "Share AutoGPT",
        amount: 1,
        details: "Share AutoGPT with others",
        action: true,
      },
      {
        id: "COMPLETED_TASK" as any,
        name: "Already done",
        amount: 1,
        details: "This task is already completed",
        action: true,
      },
    ],
  },
];

describe("TaskGroups", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    cleanup();
  });

  test("renders group names and task names", () => {
    render(<TaskGroups groups={groups} />);

    expect(screen.getByText("Getting Started")).toBeDefined();
    expect(screen.getByText("Visit the Copilot")).toBeDefined();
    expect(screen.getByText("Share AutoGPT")).toBeDefined();
  });

  test("renders task details", () => {
    render(<TaskGroups groups={groups} />);

    expect(
      screen.getByText("Open the Copilot to start your AI automation journey"),
    ).toBeDefined();
  });

  test("renders task reward amounts", () => {
    render(<TaskGroups groups={groups} />);

    expect(screen.getByText("$5.00")).toBeDefined();
  });

  test("calls onTaskClick when an actionable incomplete task is clicked", () => {
    const onTaskClick = vi.fn();
    render(<TaskGroups groups={groups} onTaskClick={onTaskClick} />);

    fireEvent.click(screen.getByText("Share AutoGPT"));

    expect(onTaskClick).toHaveBeenCalledWith("SHARE_PLATFORM");
  });

  test("does not call onTaskClick for non-action tasks", () => {
    const onTaskClick = vi.fn();
    render(<TaskGroups groups={groups} onTaskClick={onTaskClick} />);

    fireEvent.click(screen.getByText("Visit the Copilot"));

    expect(onTaskClick).not.toHaveBeenCalled();
  });

  test("does not call onTaskClick for completed action tasks", () => {
    const onTaskClick = vi.fn();
    render(<TaskGroups groups={groups} onTaskClick={onTaskClick} />);

    fireEvent.click(screen.getByText("Already done"));

    expect(onTaskClick).not.toHaveBeenCalled();
  });

  test("renders completed count", () => {
    render(<TaskGroups groups={groups} />);

    expect(screen.getByText(/1 of 3/)).toBeDefined();
  });

  test("renders Hidden Tasks section", () => {
    render(<TaskGroups groups={groups} />);

    expect(screen.getByText("Hidden Tasks")).toBeDefined();
  });
});
