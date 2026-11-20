import { render, screen } from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SettingsBillingPage from "../page";
import { mockBilling } from "./trial-plan-fixtures";

const mockSearchParams = vi.hoisted(() => ({
  current: new URLSearchParams(),
}));
const mockReplace = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useSearchParams: () => mockSearchParams.current,
  useRouter: () => ({
    push: vi.fn(),
    replace: mockReplace,
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/settings/billing",
  useParams: () => ({}),
}));

vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => true,
  useBillingStatus: () => ({ isBillingEnabled: true, isResolved: true }),
}));

beforeEach(() => setTrialUser());
afterEach(() => {
  setTrialUser(null);
  mockSearchParams.current = new URLSearchParams();
  mockReplace.mockReset();
});

describe("arriving from the canceled-trial popup's Subscribe now", () => {
  it("opens the trial plan's confirmation and drops the marker", async () => {
    mockSearchParams.current = new URLSearchParams({ subscribe: "trial-plan" });
    mockBilling();
    render(<SettingsBillingPage />);

    expect(
      await screen.findByRole("dialog", { name: "Start Pro today?" }),
    ).toBeDefined();
    expect(mockReplace).toHaveBeenCalledWith("/settings/billing");
  });

  it("opens nothing once a paid plan has replaced the trial", async () => {
    mockSearchParams.current = new URLSearchParams({ subscribe: "trial-plan" });
    const { state } = mockBilling(
      trialResponse({
        active: false,
        status: "canceled",
        cancel_at_period_end: true,
      }),
    );
    state.subscription = { ...state.subscription, tier: "PRO" };
    render(<SettingsBillingPage />);

    // The plan card reads "Active" only once the PRO subscription has
    // loaded, so nothing below can pass on a response still in flight.
    expect(await screen.findByText("Active")).toBeDefined();
    expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull();
    expect(
      screen.queryByRole("dialog", { name: "Start Pro today?" }),
    ).toBeNull();
    expect(mockReplace).not.toHaveBeenCalledWith("/settings/billing");
  });
});
