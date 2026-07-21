import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import SettingsBillingPage from "../billing/page";
import SettingsOAuthAppsPage from "../oauth-apps/page";

describe("Settings billing page", () => {
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
