import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import {
  UserOnboarding,
  WebSocketNotification,
} from "@/lib/autogpt-server-api";
import userEvent from "@testing-library/user-event";
import { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Wallet } from "../Wallet";

const confettiMock = vi.hoisted(() => vi.fn());
const fetchCreditsMock = vi.hoisted(() => vi.fn());
const updateStateMock = vi.hoisted(() => vi.fn());
const connectWebSocketMock = vi.hoisted(() => vi.fn());
const detachMessageMock = vi.hoisted(() => vi.fn());
const onWebSocketMessageMock = vi.hoisted(() => vi.fn());

const creditsState = vi.hoisted(() => ({ credits: 979 as number | null }));
const onboardingState = vi.hoisted(() => ({
  state: null as UserOnboarding | null,
}));

vi.mock("canvas-confetti", () => ({ default: confettiMock }));

vi.mock("@/hooks/useCredits", () => ({
  default: () => ({
    credits: creditsState.credits,
    fetchCredits: fetchCreditsMock,
    formatCredits: (credit: number | null) =>
      credit === null ? "-" : `$${(credit / 100).toFixed(2)}`,
    requestTopUp: vi.fn(),
    refundTopUp: vi.fn(),
    autoTopUpConfig: null,
    fetchAutoTopUpConfig: vi.fn(),
    updateAutoTopUpConfig: vi.fn(),
    transactionHistory: { transactions: [], next_transaction_time: null },
    fetchTransactionHistory: vi.fn(),
    refundRequests: [],
    fetchRefundRequests: vi.fn(),
  }),
}));

// The real provider opens its own "notification" subscription and drives
// onboarding redirects — neither is under test here, so stub it out entirely.
vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => children,
  useOnboarding: () => ({
    state: onboardingState.state,
    updateState: updateStateMock,
    step: 1,
    setStep: vi.fn(),
    completeStep: vi.fn(),
  }),
}));

// The identity has to be stable: the real context hands out one client, and
// the websocket effect is keyed on it.
const backendAPI = vi.hoisted(() => ({}) as Record<string, unknown>);

vi.mock("@/lib/autogpt-server-api/context", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/lib/autogpt-server-api/context")>();
  return { ...actual, useBackendAPI: () => backendAPI };
});

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: vi.fn(() => true) };
});

function buildOnboarding(
  overrides: Partial<UserOnboarding> = {},
): UserOnboarding {
  return {
    completedSteps: ["VISIT_COPILOT"],
    walletShown: false,
    notified: [],
    rewardedFor: [],
    usageReason: null,
    integrations: [],
    otherIntegrations: null,
    selectedStoreListingVersionId: null,
    agentInput: null,
    onboardingAgentExecutionId: null,
    lastRunAt: null,
    consecutiveRunDays: 0,
    agentRuns: 0,
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  onWebSocketMessageMock.mockReturnValue(detachMessageMock);
  backendAPI.onWebSocketMessage = onWebSocketMessageMock;
  backendAPI.connectWebSocket = connectWebSocketMock;
  creditsState.credits = 979;
  onboardingState.state = buildOnboarding();

  // happy-dom has no layout, so the wallet button measures 0×0 and the
  // confetti burst would bail out before firing.
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockReturnValue({
    ...new DOMRect(),
    top: 10,
    left: 20,
    width: 100,
    height: 40,
  });
});

describe("Wallet", () => {
  it("renders nothing until credits and onboarding state are both available", () => {
    creditsState.credits = null;
    const { container } = render(<Wallet />);

    expect(container.innerHTML).toBe("");
  });

  it("shows the balance and the unclaimed-reward dot in the classic navbar", () => {
    render(<Wallet />);

    expect(screen.getByText("$9.79")).toBeDefined();
    expect(screen.getByText("Earn credits")).toBeDefined();
    expect(screen.getByText(/1 of 8 rewards claimed/)).toBeDefined();
  });

  it("hides the reward dot and tooltip in compact mode", () => {
    render(<Wallet compact />);

    expect(screen.getByText("$9.79")).toBeDefined();
    expect(screen.queryByText(/rewards claimed/)).toBeNull();
  });

  it("opens the full panel and marks the wallet as shown", async () => {
    render(<Wallet />);

    await userEvent.click(screen.getByRole("button"));

    expect(await screen.findByText("Automation Credits")).toBeDefined();
    expect(screen.getByText(/Complete the following tasks/)).toBeDefined();
    expect(updateStateMock).toHaveBeenCalledWith({ walletShown: true });
    expect(fetchCreditsMock).toHaveBeenCalled();
  });

  it("opens the compact panel and swaps it for the add-credits dialog", async () => {
    render(<Wallet compact />);

    await userEvent.click(screen.getByRole("button"));

    expect(await screen.findByText("Automation credits")).toBeDefined();

    await userEvent.click(screen.getByRole("button", { name: "Add credits" }));

    expect(await screen.findByText("Add automation credits")).toBeDefined();
    expect(screen.queryByText("Earn credits")).toBeNull();

    await userEvent.keyboard("{Escape}");

    await waitFor(() => {
      expect(screen.queryByText("Add automation credits")).toBeNull();
    });
  });

  it("flashes the balance when the credit total changes", () => {
    const { container, rerender } = render(<Wallet />);
    const overlay = container.querySelector(".bg-violet-400");

    expect(overlay?.className).toContain("opacity-0");

    creditsState.credits = 1479;
    rerender(<Wallet />);

    expect(container.querySelector(".bg-violet-400")?.className).toContain(
      "opacity-50",
    );
    expect(screen.getByText("$14.79")).toBeDefined();
  });

  it("does not mark the wallet as shown again once it has been seen", async () => {
    onboardingState.state = buildOnboarding({ walletShown: true });
    render(<Wallet />);

    await userEvent.click(screen.getByRole("button"));

    expect(updateStateMock).not.toHaveBeenCalledWith({ walletShown: true });
  });
});

describe("Wallet onboarding notifications", () => {
  // Mounting <Wallet /> also mounts <TaskGroups />, whose celebration effect
  // schedules its own confetti burst on a 300ms timer. That timer outlives the
  // test that armed it and lands in a later one, so every assertion on
  // confettiMock has to start from a clean slate after the render — otherwise
  // the count depends on how fast the machine runs the suite.
  function emit(notification: WebSocketNotification) {
    const handler = onWebSocketMessageMock.mock.calls.at(-1)?.[1] as (
      n: WebSocketNotification,
    ) => void;
    handler(notification);
  }

  it("subscribes once and keeps the connection across onboarding updates", async () => {
    const { rerender } = render(<Wallet />);

    expect(onWebSocketMessageMock).toHaveBeenCalledTimes(1);
    expect(connectWebSocketMock).toHaveBeenCalledTimes(1);

    onboardingState.state = buildOnboarding({
      completedSteps: ["VISIT_COPILOT", "SCHEDULE_AGENT"],
    });
    rerender(<Wallet />);

    await waitFor(() => {
      expect(screen.getByText(/2 of 8 rewards claimed/)).toBeDefined();
    });
    expect(onWebSocketMessageMock).toHaveBeenCalledTimes(1);
    expect(connectWebSocketMock).toHaveBeenCalledTimes(1);
    expect(detachMessageMock).not.toHaveBeenCalled();
  });

  it("refreshes credits and fires confetti when a tracked step completes", () => {
    render(<Wallet />);
    fetchCreditsMock.mockClear();
    confettiMock.mockClear();

    emit({ type: "onboarding", event: "step_completed", step: "RUN_3_DAYS" });

    expect(fetchCreditsMock).toHaveBeenCalledTimes(1);
    expect(confettiMock).toHaveBeenCalledTimes(2);
  });

  it("ignores notifications that are not completed onboarding steps", () => {
    render(<Wallet />);
    fetchCreditsMock.mockClear();
    confettiMock.mockClear();

    emit({ type: "onboarding", event: "increment_runs", step: null });
    emit({ type: "execution", event: "step_completed" });

    expect(fetchCreditsMock).not.toHaveBeenCalled();
    expect(confettiMock).not.toHaveBeenCalled();
  });

  it("refreshes credits without confetti for steps outside the task groups", () => {
    render(<Wallet />);
    fetchCreditsMock.mockClear();
    confettiMock.mockClear();

    emit({
      type: "onboarding",
      event: "step_completed",
      step: "CONGRATS",
    });

    expect(fetchCreditsMock).toHaveBeenCalledTimes(1);
    expect(confettiMock).not.toHaveBeenCalled();
  });

  it("detaches the listener on unmount", () => {
    const { unmount } = render(<Wallet />);

    unmount();

    expect(detachMessageMock).toHaveBeenCalledTimes(1);
  });
});
