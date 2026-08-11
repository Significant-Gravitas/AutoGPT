import type { NotificationPreference } from "@/app/api/__generated__/models/notificationPreference";
import {
  ChartIncreaseIcon,
  Coins01Icon,
  Notification03Icon,
  Store01Icon,
} from "@hugeicons/core-free-icons";

export const EASE_OUT = [0.16, 1, 0.3, 1] as const;
export const EASE_IOS = [0.32, 0.72, 0, 1] as const;
export const EASE_IN_OUT = [0.4, 0, 0.2, 1] as const;

export type BriefingFrequency = "DAILY" | "WEEKLY" | "MONTHLY" | "OFF";

/**
 * The volume knob from the Briefing footer: a frequency plus two switches.
 * Billing and account messages are service mail and are deliberately absent —
 * they are sent regardless of what is set here.
 */
export interface NotificationSettings {
  briefingFrequency: BriefingFrequency;
  alertsEnabled: boolean;
  storeVerdictsEnabled: boolean;
}

export interface PreferencesFormState {
  timezone: string;
  notifications: NotificationSettings;
}

export const BRIEFING_OPTIONS: { value: BriefingFrequency; label: string }[] = [
  { value: "DAILY", label: "Daily" },
  { value: "WEEKLY", label: "Weekly" },
  { value: "MONTHLY", label: "Monthly" },
  { value: "OFF", label: "Off" },
];

export const NOTIFICATIONS_FALLBACK_ICON = Notification03Icon;
export const BRIEFING_ICON = ChartIncreaseIcon;
export const ALERTS_ICON = Coins01Icon;
export const VERDICT_ICON = Store01Icon;

/**
 * The `?f=` values the Briefing footer links use. "alerts" and "off" both turn
 * the digest off; they differ in whether alerts survive.
 */
export function settingsFromFooterLink(
  value: string | null,
  current: NotificationSettings,
): NotificationSettings | null {
  switch (value) {
    case "daily":
      return { ...current, briefingFrequency: "DAILY" };
    case "weekly":
      return { ...current, briefingFrequency: "WEEKLY" };
    case "monthly":
      return { ...current, briefingFrequency: "MONTHLY" };
    case "alerts":
      return { ...current, briefingFrequency: "OFF", alertsEnabled: true };
    case "off":
      return { ...current, briefingFrequency: "OFF", alertsEnabled: false };
    default:
      return null;
  }
}

export function preferencesToSettings(
  preferences: NotificationPreference,
): NotificationSettings {
  return {
    briefingFrequency: (preferences.briefing_frequency ??
      "WEEKLY") as BriefingFrequency,
    alertsEnabled: preferences.alerts_enabled ?? true,
    storeVerdictsEnabled: preferences.store_verdicts_enabled ?? true,
  };
}

export function detectBrowserTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
  } catch {
    return "UTC";
  }
}

export function isFormDirty(
  initial: PreferencesFormState,
  current: PreferencesFormState,
): boolean {
  const parts = dirtyKinds(initial, current);
  return parts.timezone || parts.notifications;
}

export function dirtyKinds(
  initial: PreferencesFormState,
  current: PreferencesFormState,
): { timezone: boolean; notifications: boolean } {
  return {
    timezone: initial.timezone !== current.timezone,
    notifications:
      initial.notifications.briefingFrequency !==
        current.notifications.briefingFrequency ||
      initial.notifications.alertsEnabled !==
        current.notifications.alertsEnabled ||
      initial.notifications.storeVerdictsEnabled !==
        current.notifications.storeVerdictsEnabled,
  };
}

const TIMEZONE_LIST: { value: string; label: string }[] = [
  { value: "UTC", label: "UTC (Coordinated Universal Time)" },
  { value: "America/Los_Angeles", label: "Los Angeles (US - Pacific)" },
  { value: "America/Denver", label: "Denver (US - Mountain)" },
  { value: "America/Chicago", label: "Chicago (US - Central)" },
  { value: "America/New_York", label: "New York (US - Eastern)" },
  { value: "America/Toronto", label: "Toronto (Canada - Eastern)" },
  { value: "America/Mexico_City", label: "Mexico City (Mexico)" },
  { value: "America/Sao_Paulo", label: "São Paulo (Brazil)" },
  { value: "America/Buenos_Aires", label: "Buenos Aires (Argentina)" },
  { value: "America/Bogota", label: "Bogotá (Colombia)" },
  { value: "America/Lima", label: "Lima (Peru)" },
  { value: "America/Santiago", label: "Santiago (Chile)" },
  { value: "Europe/London", label: "London (UK)" },
  { value: "Europe/Dublin", label: "Dublin (Ireland)" },
  { value: "Europe/Lisbon", label: "Lisbon (Portugal)" },
  { value: "Europe/Madrid", label: "Madrid (Spain)" },
  { value: "Europe/Paris", label: "Paris (France)" },
  { value: "Europe/Amsterdam", label: "Amsterdam (Netherlands)" },
  { value: "Europe/Brussels", label: "Brussels (Belgium)" },
  { value: "Europe/Berlin", label: "Berlin (Germany)" },
  { value: "Europe/Zurich", label: "Zurich (Switzerland)" },
  { value: "Europe/Rome", label: "Rome (Italy)" },
  { value: "Europe/Vienna", label: "Vienna (Austria)" },
  { value: "Europe/Prague", label: "Prague (Czechia)" },
  { value: "Europe/Warsaw", label: "Warsaw (Poland)" },
  { value: "Europe/Stockholm", label: "Stockholm (Sweden)" },
  { value: "Europe/Oslo", label: "Oslo (Norway)" },
  { value: "Europe/Copenhagen", label: "Copenhagen (Denmark)" },
  { value: "Europe/Helsinki", label: "Helsinki (Finland)" },
  { value: "Europe/Athens", label: "Athens (Greece)" },
  { value: "Europe/Istanbul", label: "Istanbul (Türkiye)" },
  { value: "Europe/Moscow", label: "Moscow (Russia)" },
  { value: "Africa/Cairo", label: "Cairo (Egypt)" },
  { value: "Africa/Lagos", label: "Lagos (Nigeria)" },
  { value: "Africa/Nairobi", label: "Nairobi (Kenya)" },
  { value: "Africa/Johannesburg", label: "Johannesburg (South Africa)" },
  { value: "Asia/Dubai", label: "Dubai (UAE)" },
  { value: "Asia/Tehran", label: "Tehran (Iran)" },
  { value: "Asia/Karachi", label: "Karachi (Pakistan)" },
  { value: "Asia/Kolkata", label: "Kolkata (India)" },
  { value: "Asia/Dhaka", label: "Dhaka (Bangladesh)" },
  { value: "Asia/Bangkok", label: "Bangkok (Thailand)" },
  { value: "Asia/Jakarta", label: "Jakarta (Indonesia)" },
  { value: "Asia/Singapore", label: "Singapore" },
  { value: "Asia/Manila", label: "Manila (Philippines)" },
  { value: "Asia/Hong_Kong", label: "Hong Kong" },
  { value: "Asia/Shanghai", label: "Shanghai (China)" },
  { value: "Asia/Taipei", label: "Taipei (Taiwan)" },
  { value: "Asia/Seoul", label: "Seoul (South Korea)" },
  { value: "Asia/Tokyo", label: "Tokyo (Japan)" },
  { value: "Australia/Perth", label: "Perth (Australia - West)" },
  { value: "Australia/Adelaide", label: "Adelaide (Australia - Central)" },
  { value: "Australia/Brisbane", label: "Brisbane (Australia - East)" },
  { value: "Australia/Sydney", label: "Sydney (Australia - East)" },
  { value: "Pacific/Auckland", label: "Auckland (New Zealand)" },
  { value: "Pacific/Honolulu", label: "Honolulu (US - Hawaii)" },
];

export const TIMEZONES = TIMEZONE_LIST;

export function findTimezoneLabel(value: string): string {
  return TIMEZONES.find((t) => t.value === value)?.label ?? value;
}
