import { describe, expect, test, vi } from "vitest";

import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { NotificationsCard } from "../NotificationsCard";

describe("NotificationsCard", () => {
  test("renders the email volume controls and reports every change", async () => {
    const onBriefingFrequencyChange = vi.fn();
    const onAlertsChange = vi.fn();
    const onStoreVerdictsChange = vi.fn();

    render(
      <NotificationsCard
        values={{
          briefingFrequency: "WEEKLY",
          alertsEnabled: true,
          storeVerdictsEnabled: false,
        }}
        onBriefingFrequencyChange={onBriefingFrequencyChange}
        onAlertsChange={onAlertsChange}
        onStoreVerdictsChange={onStoreVerdictsChange}
      />,
    );

    fireEvent.click(screen.getByRole("combobox", { name: "Briefing" }));
    fireEvent.click(await screen.findByRole("option", { name: "Monthly" }));
    fireEvent.click(screen.getByRole("switch", { name: "Alerts" }));
    fireEvent.click(
      screen.getByRole("switch", { name: "Marketplace reviews" }),
    );

    expect(onBriefingFrequencyChange).toHaveBeenCalledWith("MONTHLY");
    expect(onAlertsChange).toHaveBeenCalledWith(false);
    expect(onStoreVerdictsChange).toHaveBeenCalledWith(true);
  });
});
