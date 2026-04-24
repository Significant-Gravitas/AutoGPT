import {
  render,
  screen,
  cleanup,
  fireEvent,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { UsagePanelContent } from "../UsagePanelContent";
import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";

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
    dailyPercent: number | null;
    weeklyPercent: number | null;
    tier: string;
    resetCost: number;
  }> = {},
): CoPilotUsagePublic {
  const {
    dailyPercent = 5,
    weeklyPercent = 4,
    tier = "BASIC",
    resetCost = 100,
  } = overrides;
  const future = new Date(Date.now() + 3600 * 1000).toISOString();
  return {
    daily:
      dailyPercent === null
        ? null
        : { percent_used: dailyPercent, resets_at: future },
    weekly:
      weeklyPercent === null
        ? null
        : { percent_used: weeklyPercent, resets_at: future },
    tier,
    reset_cost: resetCost,
  } as CoPilotUsagePublic;
}

describe("UsagePanelContent", () => {
  it("renders 'No usage limits configured' when both windows are null", () => {
    render(
      <UsagePanelContent
        usage={makeUsage({ dailyPercent: null, weeklyPercent: null })}
      />,
    );
    expect(screen.getByText("No usage limits configured")).toBeDefined();
  });

  it("renders the reset button when daily limit is exhausted", () => {
    render(
      <UsagePanelContent
        usage={makeUsage({ dailyPercent: 100, resetCost: 50 })}
      />,
    );
    expect(screen.getByText(/Reset daily limit/)).toBeDefined();
  });

  it("does not render the reset button when weekly limit is also exhausted", () => {
    render(
      <UsagePanelContent
        usage={makeUsage({
          dailyPercent: 100,
          weeklyPercent: 100,
          resetCost: 50,
        })}
      />,
    );
    expect(screen.queryByText(/Reset daily limit/)).toBeNull();
  });

  it("calls resetUsage when the reset button is clicked", () => {
    render(
      <UsagePanelContent
        usage={makeUsage({ dailyPercent: 100, resetCost: 50 })}
      />,
    );
    fireEvent.click(screen.getByText(/Reset daily limit/));
    expect(mockResetUsage).toHaveBeenCalled();
  });

  it("renders 'Add credits' link when insufficient credits", () => {
    render(
      <UsagePanelContent
        usage={makeUsage({ dailyPercent: 100, resetCost: 50 })}
        hasInsufficientCredits={true}
        isBillingEnabled={true}
      />,
    );
    expect(screen.getByText("Add credits to reset")).toBeDefined();
  });

  it("renders percent used in the usage bar", () => {
    render(<UsagePanelContent usage={makeUsage({ dailyPercent: 25 })} />);
    expect(screen.getByText("25% used")).toBeDefined();
  });

  it("renders '<1% used' when usage is greater than 0 but rounds to 0", () => {
    render(<UsagePanelContent usage={makeUsage({ dailyPercent: 0.3 })} />);
    expect(screen.getByText("<1% used")).toBeDefined();
  });
});
