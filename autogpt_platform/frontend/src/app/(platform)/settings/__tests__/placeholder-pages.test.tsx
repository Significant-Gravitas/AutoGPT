import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import SettingsBillingPage from "../billing/page";
import SettingsOAuthAppsPage from "../oauth-apps/page";

let mockIsBillingEnabled = true;
vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => mockIsBillingEnabled,
  useBillingStatus: () => ({
    isBillingEnabled: mockIsBillingEnabled,
    isResolved: true,
  }),
}));

describe("Settings billing page", () => {
  afterEach(() => {
    mockIsBillingEnabled = true;
  });

  it("is not found on a self-hosted install without payments", () => {
    mockIsBillingEnabled = false;

    vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => render(<SettingsBillingPage />)).toThrow(
      "NEXT_HTTP_ERROR_FALLBACK;404",
    );
  });

  it("renders title and billing tabs", () => {
    render(<SettingsBillingPage />);

    expect(screen.getByRole("heading", { name: "Billing" })).toBeDefined();
    expect(screen.getByRole("tab", { name: "Subscription" })).toBeDefined();
    expect(
      screen.getByRole("tab", { name: "Automation Credits" }),
    ).toBeDefined();
  });
});

describe("Settings v2 placeholder pages", () => {
  it("OAuth Apps renders title and coming soon body", () => {
    render(<SettingsOAuthAppsPage />);

    expect(screen.getByText("OAuth Apps")).toBeDefined();
    expect(screen.getByText(/coming soon/i)).toBeDefined();
  });
});
