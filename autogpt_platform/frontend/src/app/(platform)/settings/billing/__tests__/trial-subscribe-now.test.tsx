import { render, screen, waitFor } from "@/tests/integrations/test-utils";
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

  it("opens nothing once the trial is no longer cancel-pending", async () => {
    mockSearchParams.current = new URLSearchParams({ subscribe: "trial-plan" });
    const { state } = mockBilling(trialResponse());
    state.subscription = { ...state.subscription, tier: "PRO" };
    render(<SettingsBillingPage />);

    await waitFor(() =>
      expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull(),
    );
    expect(
      screen.queryByRole("dialog", { name: "Start Pro today?" }),
    ).toBeNull();
  });
});
