import { describe, expect, test, vi } from "vitest";
import { render, screen } from "@testing-library/react";

const updateState = vi.fn();
let mockCompletedSteps: string[] = [];

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: React.ReactNode }) => children,
  useOnboarding: () => ({
    state: {
      completedSteps: mockCompletedSteps,
      consecutiveRunDays: 0,
      agentRuns: 0,
      walletShown: true,
    },
    updateState,
  }),
}));

vi.mock("@/hooks/useCredits", () => ({
  default: () => ({
    credits: 100,
    formatCredits: (amount: number) => `$${amount}`,
    fetchCredits: vi.fn(),
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ENABLE_PLATFORM_PAYMENT: "ENABLE_PLATFORM_PAYMENT" },
  useGetFlag: () => false,
}));

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => ({
    onWebSocketMessage: () => () => {},
    connectWebSocket: () => {},
  }),
}));

import { Wallet } from "../Wallet";

describe("Wallet", () => {
  test("counts ONBOARDING_COMPLETE as a claimed first-win reward", async () => {
    // The first-win "Complete onboarding" task is keyed on ONBOARDING_COMPLETE
    // (renamed from VISIT_COPILOT). With that step completed, exactly one of the
    // eight tasks should register as claimed — this would read "0 of 8" if the
    // task id were still the retired value.
    mockCompletedSteps = ["ONBOARDING_COMPLETE"];

    render(<Wallet />);

    const claimed = await screen.findByText(/rewards claimed/);
    expect(claimed.textContent).toContain("1 of 8 rewards claimed");
  });
});
