import { act, render, screen, waitFor } from "@/tests/integrations/test-utils";
import {
  UserOnboarding,
  WebSocketNotification,
} from "@/lib/autogpt-server-api";
import userEvent from "@testing-library/user-event";
import { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import { server } from "@/mocks/mock-server";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";

import { Wallet } from "../Wallet";
import { TaskGroups } from "../components/WalletTaskGroups";
import { getTaskGroups } from "../helpers";

// Noon on Sep 17 in the runner's own time zone: the trial date is shown in the
// viewer's time zone, so a fixed UTC instant lands on Sep 18 east of UTC+9.
const SEP_17_LOCAL_NOON = new Date(2030, 8, 17, 12);

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
    completedSteps: ["ONBOARDING_COMPLETE"],
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

afterEach(() => {
  if (vi.isFakeTimers()) {
    vi.clearAllTimers();
    vi.useRealTimers();
  }
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

  it("shows an accessible icon-only trigger in compact mode", () => {
    render(<Wallet compact />);

    const trigger = screen.getByRole("button", { name: "Usage and credits" });
    expect(trigger.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByText("$9.79")).toBeNull();
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
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "Usage and credits" }),
      );
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

describe("Wallet timer cleanup", () => {
  it("cancels delayed task confetti when task groups unmount", () => {
    vi.useFakeTimers();
    const { unmount } = render(
      <TaskGroups groups={getTaskGroups(onboardingState.state)} />,
    );
    expect(vi.getTimerCount()).toBeGreaterThan(0);

    unmount();
    vi.runOnlyPendingTimers();

    expect(confettiMock).not.toHaveBeenCalled();
  });

  it("cancels the pending balance flash when the wallet unmounts", () => {
    vi.useFakeTimers();
    const { rerender, unmount } = render(<Wallet compact />);

    creditsState.credits = 1479;
    rerender(<Wallet compact />);
    expect(vi.getTimerCount()).toBeGreaterThan(0);

    unmount();

    expect(vi.getTimerCount()).toBe(0);
  });

  it("resets the balance flash after credits briefly become unavailable", () => {
    vi.useFakeTimers();
    const { container, rerender } = render(<Wallet compact />);

    creditsState.credits = null;
    rerender(<Wallet compact />);
    creditsState.credits = 1479;
    rerender(<Wallet compact />);

    expect(container.querySelector(".bg-violet-400")?.className).toContain(
      "opacity-50",
    );

    act(() => vi.advanceTimersByTime(300));

    expect(container.querySelector(".bg-violet-400")?.className).toContain(
      "opacity-0",
    );
  });
});

describe("Wallet onboarding notifications", () => {
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
      completedSteps: ["ONBOARDING_COMPLETE", "SCHEDULE_AGENT"],
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

// SECRT-3133: a trial user saw chat usage, automation credits and earn credits
// with nothing saying which one an image comes out of, or whether the card
// they had just entered would be charged.
describe("Wallet trial line and explainer", () => {
  afterEach(() => setTrialUser(null));

  function mockTrial(overrides: Parameters<typeof trialResponse>[0] = {}) {
    setTrialUser();
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(trialResponse(overrides)),
    );
  }

  async function openWallet() {
    render(<Wallet compact />);
    await userEvent.click(
      screen.getByRole("button", { name: "Usage and credits" }),
    );
    return screen.findByRole("dialog", { name: "Usage and credits" });
  }

  it("names the day the plan starts in one line during a trial", async () => {
    mockTrial({ ends_at: SEP_17_LOCAL_NOON });
    const dialog = await openWallet();

    const line = await screen.findByText("Your plan starts Sep 17");
    // Focus lands on the first button, so the line is the dialog's
    // description: a screen reader announces it as the wallet opens.
    const description = document.getElementById(
      dialog.getAttribute("aria-describedby") ?? "",
    );
    expect(description?.contains(line)).toBe(true);
    expect(screen.queryByText(/charged/)).toBeNull();
  });

  it("explains usage and credits in a dialog opened from the wallet", async () => {
    mockTrial();
    await openWallet();

    await userEvent.click(
      await screen.findByRole("button", { name: "How it works" }),
    );

    expect(
      await screen.findByRole("dialog", { name: "How usage and credits work" }),
    ).toBeDefined();
    expect(screen.getByText("What does chat usage cover?")).toBeDefined();
    expect(screen.getByText("When is my card charged?")).toBeDefined();
    expect(
      screen.queryByRole("dialog", { name: "Usage and credits" }),
    ).toBeNull();
  });

  // Sentry: the explainer opens from inside the popover, which unmounts, so
  // closing it left focus on <body> instead of the wallet button.
  it("returns focus to the wallet button when the explainer closes", async () => {
    mockTrial();
    await openWallet();

    await userEvent.click(
      await screen.findByRole("button", { name: "How it works" }),
    );
    await screen.findByRole("dialog", { name: "How usage and credits work" });
    await userEvent.keyboard("{Escape}");

    await waitFor(() => {
      expect(
        screen.queryByRole("dialog", { name: "How usage and credits work" }),
      ).toBeNull();
      expect(document.activeElement).toBe(
        screen.getByRole("button", { name: "Usage and credits" }),
      );
    });
  });

  it("names the trial's end instead of a plan once the trial is canceled", async () => {
    mockTrial({
      cancel_at_period_end: true,
      ends_at: SEP_17_LOCAL_NOON,
    });
    await openWallet();

    expect(await screen.findByText("Your trial ends Sep 17")).toBeDefined();
    expect(screen.queryByText(/Your plan starts/)).toBeNull();
  });

  it("shows no trial line once the trial is over", async () => {
    mockTrial({ active: false });
    await openWallet();

    expect(await screen.findByText("Automation credits")).toBeDefined();
    await waitFor(() => {
      expect(screen.queryByText(/Your plan starts|Your trial ends/)).toBeNull();
    });
  });

  it("marks the wallet until it has been opened once", () => {
    const { container, rerender } = render(<Wallet compact />);
    expect(
      container.querySelector('[data-testid="wallet-unseen-dot"]'),
    ).not.toBeNull();

    onboardingState.state = buildOnboarding({ walletShown: true });
    rerender(<Wallet compact />);

    expect(
      container.querySelector('[data-testid="wallet-unseen-dot"]'),
    ).toBeNull();
  });
});
