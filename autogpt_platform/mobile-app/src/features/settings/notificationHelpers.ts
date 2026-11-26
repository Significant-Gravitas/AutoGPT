// No native imports, so tests can load this without a phone.
import type { BriefingFrequency, NotificationPreferences } from "../../lib/api/preferences";

const NOT_SET = "not-set";

export const BRIEFING_OPTIONS: { value: BriefingFrequency; label: string }[] = [
  { value: "DAILY", label: "Daily" },
  { value: "WEEKLY", label: "Weekly" },
  { value: "MONTHLY", label: "Monthly" },
  { value: "OFF", label: "Off" },
];

/** The server writes every field, so a change goes out with the rest as saved. */
export function withChange(
  saved: NotificationPreferences,
  change: Partial<NotificationPreferences>,
): NotificationPreferences {
  return { ...saved, ...change };
}

/** The phone's zone, when there is one and the account isn't on it already. */
export function timezoneOffer(serverZone: string, phoneZone: string) {
  if (!phoneZone || phoneZone === serverZone) return null;
  return phoneZone;
}

export function timezoneLabel(zone: string) {
  return zone === NOT_SET ? "Not set" : zone;
}

export function phoneTimezone() {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "";
  } catch {
    return "";
  }
}
