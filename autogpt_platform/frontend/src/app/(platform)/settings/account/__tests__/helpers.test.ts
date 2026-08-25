import { describe, expect, it, test } from "vitest";

import type { NotificationPreference } from "@/app/api/__generated__/models/notificationPreference";

import {
  BRIEFING_OPTIONS,
  describeFooterChoice,
  dirtyKinds,
  findTimezoneLabel,
  isFormDirty,
  preferencesToSettings,
  settingsFromFooterLink,
  type NotificationSettings,
  type PreferencesFormState,
} from "../helpers";

const currentSettings: NotificationSettings = {
  briefingFrequency: "WEEKLY",
  alertsEnabled: false,
  storeVerdictsEnabled: true,
};

function makeState(
  notifications: NotificationSettings = currentSettings,
  timezone = "Europe/London",
): PreferencesFormState {
  return { timezone, notifications };
}

describe("account preference helpers", () => {
  test("offers every supported briefing frequency", () => {
    expect(BRIEFING_OPTIONS.map(({ value }) => value)).toEqual([
      "DAILY",
      "WEEKLY",
      "MONTHLY",
      "OFF",
    ]);
  });

  test("maps every email footer choice to notification settings", () => {
    expect(settingsFromFooterLink("daily", currentSettings)).toEqual({
      ...currentSettings,
      briefingFrequency: "DAILY",
    });
    expect(settingsFromFooterLink("weekly", currentSettings)).toEqual({
      ...currentSettings,
      briefingFrequency: "WEEKLY",
    });
    expect(settingsFromFooterLink("monthly", currentSettings)).toEqual({
      ...currentSettings,
      briefingFrequency: "MONTHLY",
    });
    expect(settingsFromFooterLink("alerts", currentSettings)).toEqual({
      ...currentSettings,
      briefingFrequency: "OFF",
      alertsEnabled: true,
    });
    expect(settingsFromFooterLink("off", currentSettings)).toEqual({
      ...currentSettings,
      briefingFrequency: "OFF",
      alertsEnabled: false,
    });
    expect(settingsFromFooterLink(null, currentSettings)).toBeNull();
    expect(settingsFromFooterLink("unsupported", currentSettings)).toBeNull();
  });

  test("uses safe defaults for omitted notification preferences", () => {
    const preferences = {
      user_id: "user-1",
      email: "user@example.com",
    } satisfies NotificationPreference;

    expect(preferencesToSettings(preferences)).toEqual({
      briefingFrequency: "WEEKLY",
      alertsEnabled: true,
      storeVerdictsEnabled: true,
    });
  });

  test("preserves notification preferences returned by the API", () => {
    const preferences = {
      user_id: "user-1",
      email: "user@example.com",
      briefing_frequency: "MONTHLY",
      alerts_enabled: false,
      store_verdicts_enabled: false,
    } satisfies NotificationPreference;

    expect(preferencesToSettings(preferences)).toEqual({
      briefingFrequency: "MONTHLY",
      alertsEnabled: false,
      storeVerdictsEnabled: false,
    });
  });

  test("reports which preference groups have unsaved changes", () => {
    const initial = makeState();
    const briefingChanged = makeState({
      ...currentSettings,
      briefingFrequency: "DAILY",
    });
    const alertsChanged = makeState({
      ...currentSettings,
      alertsEnabled: true,
    });
    const verdictsChanged = makeState({
      ...currentSettings,
      storeVerdictsEnabled: false,
    });

    expect(dirtyKinds(initial, initial)).toEqual({
      timezone: false,
      notifications: false,
    });
    expect(
      dirtyKinds(initial, makeState(currentSettings, "Asia/Tokyo")),
    ).toEqual({
      timezone: true,
      notifications: false,
    });
    expect(dirtyKinds(initial, briefingChanged).notifications).toBe(true);
    expect(dirtyKinds(initial, alertsChanged).notifications).toBe(true);
    expect(dirtyKinds(initial, verdictsChanged).notifications).toBe(true);
    expect(isFormDirty(initial, initial)).toBe(false);
    expect(isFormDirty(initial, verdictsChanged)).toBe(true);
  });

  test("shows known timezone labels and preserves unknown values", () => {
    expect(findTimezoneLabel("Europe/London")).toBe("London (UK)");
    expect(findTimezoneLabel("Mars/Olympus_Mons")).toBe("Mars/Olympus_Mons");
  });
});

describe("the volume knob's footer links", () => {
  it("maps every option the Briefing footer offers", () => {
    const current = {
      briefingFrequency: "WEEKLY" as const,
      alertsEnabled: true,
      storeVerdictsEnabled: true,
    };
    // The five links in briefing.html.j2's footer, in order.
    expect(settingsFromFooterLink("daily", current)?.briefingFrequency).toBe(
      "DAILY",
    );
    expect(settingsFromFooterLink("weekly", current)?.briefingFrequency).toBe(
      "WEEKLY",
    );
    expect(settingsFromFooterLink("monthly", current)?.briefingFrequency).toBe(
      "MONTHLY",
    );

    // "Alerts only" and "Pause" both stop the digest; they differ on alerts.
    const alertsOnly = settingsFromFooterLink("alerts", current);
    expect(alertsOnly?.briefingFrequency).toBe("OFF");
    expect(alertsOnly?.alertsEnabled).toBe(true);

    const paused = settingsFromFooterLink("off", current);
    expect(paused?.briefingFrequency).toBe("OFF");
    expect(paused?.alertsEnabled).toBe(false);
  });

  it("leaves unrelated preferences alone", () => {
    const current = {
      briefingFrequency: "WEEKLY" as const,
      alertsEnabled: true,
      storeVerdictsEnabled: false,
    };
    // A frequency link must not quietly re-enable store verdicts.
    expect(settingsFromFooterLink("daily", current)?.storeVerdictsEnabled).toBe(
      false,
    );
  });

  it("ignores an unknown or absent choice", () => {
    const current = {
      briefingFrequency: "WEEKLY" as const,
      alertsEnabled: true,
      storeVerdictsEnabled: true,
    };
    expect(settingsFromFooterLink(null, current)).toBeNull();
    expect(settingsFromFooterLink("bogus", current)).toBeNull();
  });

  it("confirms the choice rather than prompting for it", () => {
    // The click already saved; the toast is a receipt, not a question.
    expect(describeFooterChoice("daily")).toMatch(/every day/i);
    expect(describeFooterChoice("alerts")).toMatch(/still get alerts/i);
    expect(describeFooterChoice("off")).toMatch(/paused/i);
    expect(describeFooterChoice(null)).toBe("Preferences updated");
  });
});
