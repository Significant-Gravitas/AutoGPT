import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import SettingsBillingPage from "../billing/page";
import SettingsOAuthAppsPage from "../oauth-apps/page";

const pages = [
  { Component: SettingsBillingPage, title: "Billing" },
  { Component: SettingsOAuthAppsPage, title: "OAuth Apps" },
];

describe("Settings v2 placeholder pages", () => {
  it.each(pages)(
    "$title renders title and coming soon body",
    ({ Component, title }) => {
      render(<Component />);

      expect(screen.getByText(title)).toBeDefined();
      expect(screen.getByText(/coming soon/i)).toBeDefined();
    },
  );
});
