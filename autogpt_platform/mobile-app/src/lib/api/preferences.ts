import { apiRequest } from "./client";

/** `BriefingFrequency` in backend/data/notifications.py. */
export type BriefingFrequency = "DAILY" | "WEEKLY" | "MONTHLY" | "OFF";

/** `NotificationPreference`, the same shape the update takes back. */
export interface NotificationPreferences {
  /** Ignored on update: the address changes only through the web's verified flow. */
  email: string;
  briefing_frequency: BriefingFrequency;
  alerts_enabled: boolean;
  store_verdicts_enabled: boolean;
  /** Product emails a day; 0 sends none. Left out, the server writes 3. */
  daily_limit: number;
}

/** An IANA zone, or "not-set" before anyone chose one. */
export interface Timezone {
  timezone: string;
}

export const preferenceKeys = {
  notifications: ["user", "preferences"] as const,
  timezone: ["user", "timezone"] as const,
};

export function getNotificationPreferences() {
  return apiRequest<NotificationPreferences>("/api/auth/user/preferences");
}

/** Every field is written, so send the saved ones back with the change. */
export function updateNotificationPreferences(prefs: NotificationPreferences) {
  return apiRequest<NotificationPreferences>("/api/auth/user/preferences", {
    method: "POST",
    body: {
      email: prefs.email,
      briefing_frequency: prefs.briefing_frequency,
      alerts_enabled: prefs.alerts_enabled,
      store_verdicts_enabled: prefs.store_verdicts_enabled,
      daily_limit: prefs.daily_limit,
    },
  });
}

export function getTimezone() {
  return apiRequest<Timezone>("/api/auth/user/timezone");
}

/** The server refuses anything that isn't an IANA zone name. */
export function updateTimezone(timezone: string) {
  return apiRequest<Timezone>("/api/auth/user/timezone", {
    method: "POST",
    body: { timezone },
  });
}
