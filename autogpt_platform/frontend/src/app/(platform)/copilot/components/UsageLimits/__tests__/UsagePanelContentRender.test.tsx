import {
  render,
  screen,
  cleanup,
  fireEvent,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { UsagePanelContent } from "../UsagePanelContent";
import type { CoPilotUsageStatus } from "@/app/api/__generated__/models/coPilotUsageStatus";

const mockResetUsage = vi.fn();
vi.mock("../../../hooks/useResetRateLimit", () => ({
  useResetRateLimit: () => ({ resetUsage: mockResetUsage, isPending: false }),
}));

afterEach(() => {
  cleanup();
  mockResetUsage.mockReset();
});

function makeUsage(
  overrides: Partial<{
    dailyUsed: number;
    dailyLimit: number;
    weeklyUsed: number;
    weeklyLimit: number;
    tier: string;
    resetCost: number;
  }> = {},
): CoPilotUsageStatus {
  const {
    dailyUsed = 500,
    dailyLimit = 10000,
    weeklyUsed = 2000,
    weeklyLimit = 50000,
    tier = "FREE",
    resetCost = 100,
  } = overrides;
  const future = new Date(Date.now() + 3600 * 1000);
  return {
    daily: { used: dailyUsed, limit: dailyLimit, resets_at: future },
    weekly: { used: weeklyUsed, limit: weeklyLimit, resets_at: future },
    tier,
    reset_cost: resetCost,
  } as CoPilotUsageStatus;
}

describe("UsagePanelContent", () => {
  it("renders 'No usage limits configured' when both limits are zero", () => {
    render(
      <UsagePanelContent
        usage={makeUsage({ dailyLimit: 0, weeklyLimit: 0 })}
      />,
    );
    expect(screen.getByText("No usage limits configured")).toBeDefined();
  });

  it("renders the reset button when daily limit is exhausted", () => {
    render(
      <UsagePanelContent
        usage={makeUsage({
          dailyUsed: 10000,
          dailyLimit: 10000,
          resetCost: 50,
        })}
      />,
    );
    expect(screen.getByText(/Reset daily limit/)).toBeDefined();
  });

  it("does not render the reset button when weekly limit is also exhausted", () => {
    render(
      <UsagePanelContent
        usage={makeUsage({
          dailyUsed: 10000,
          dailyLimit: 10000,
          weeklyUsed: 50000,
          weeklyLimit: 50000,
          resetCost: 50,
        })}
      />,
    );
    expect(screen.queryByText(/Reset daily limit/)).toBeNull();
  });

  it("calls resetUsage when the reset button is clicked", () => {
    render(
      <UsagePanelContent
        usage={makeUsage({
          dailyUsed: 10000,
          dailyLimit: 10000,
          resetCost: 50,
        })}
      />,
    );
    fireEvent.click(screen.getByText(/Reset daily limit/));
    expect(mockResetUsage).toHaveBeenCalled();
  });

  it("renders 'Add credits' link when insufficient credits", () => {
    render(
      <UsagePanelContent
        usage={makeUsage({
          dailyUsed: 10000,
          dailyLimit: 10000,
          resetCost: 50,
        })}
        hasInsufficientCredits={true}
        isBillingEnabled={true}
      />,
    );
    expect(screen.getByText("Add credits to reset")).toBeDefined();
  });
});
