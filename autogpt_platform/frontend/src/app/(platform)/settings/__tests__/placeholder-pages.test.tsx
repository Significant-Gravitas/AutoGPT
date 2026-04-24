import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import SettingsProfilePage from "../profile/page";
import SettingsCreatorDashboardPage from "../creator-dashboard/page";
import SettingsBillingPage from "../billing/page";
import SettingsIntegrationsPage from "../integrations/page";
import SettingsPreferencesPage from "../preferences/page";
import SettingsOAuthAppsPage from "../oauth-apps/page";

const pages = [
  { Component: SettingsProfilePage, title: "Profile" },
  { Component: SettingsCreatorDashboardPage, title: "Creator Dashboard" },
  { Component: SettingsBillingPage, title: "Billing" },
  { Component: SettingsIntegrationsPage, title: "Integrations" },
  { Component: SettingsPreferencesPage, title: "Settings" },
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
