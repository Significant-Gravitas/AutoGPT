import {
  BellSimpleRingingIcon,
  ChartLineUpIcon,
  CoinsIcon,
  RobotIcon,
  StorefrontIcon,
} from "@phosphor-icons/react";
import type { Icon } from "@phosphor-icons/react";

import type { NotificationPreference } from "@/app/api/__generated__/models/notificationPreference";

export const EASE_OUT = [0.16, 1, 0.3, 1] as const;
export const EASE_IOS = [0.32, 0.72, 0, 1] as const;
export const EASE_IN_OUT = [0.4, 0, 0.2, 1] as const;

export type NotificationKey =
  | "notifyOnAgentRun"
  | "notifyOnBlockExecutionFailed"
  | "notifyOnContinuousAgentError"
  | "notifyOnAgentApproved"
  | "notifyOnAgentRejected"
  | "notifyOnZeroBalance"
  | "notifyOnLowBalance"
  | "notifyOnDailySummary"
  | "notifyOnWeeklySummary"
  | "notifyOnMonthlySummary";

export type NotificationFlags = Record<NotificationKey, boolean>;

export interface PreferencesFormState {
  timezone: string;
  notifications: NotificationFlags;
}

export interface NotificationItem {
  key: NotificationKey;
  title: string;
  description: string;
}

export interface NotificationGroup {
  id: "agents" | "store" | "balance" | "summary";
  title: string;
  caption: string;
  icon: Icon;
  accent: string;
  items: NotificationItem[];
}

export const NOTIFICATION_GROUPS: NotificationGroup[] = [
  {
    id: "agents",
    title: "Agent activity",
    caption: "Heads-up when your agents do something — or stop doing it.",
    icon: RobotIcon,
    accent: "from-violet-500/15 to-violet-500/0 text-violet-700",
    items: [
      {
        key: "notifyOnAgentRun",
        title: "Run started or completed",
        description: "Hear from us when an agent kicks off or finishes a run.",
      },
      {
        key: "notifyOnBlockExecutionFailed",
        title: "Block execution failed",
        description: "Get notified the moment a block fails mid-run.",
      },
      {
        key: "notifyOnContinuousAgentError",
        title: "Continuous errors",
        description:
          "Alert me when an agent runs into the same error repeatedly.",
      },
    ],
  },
  {
    id: "store",
    title: "Marketplace",
    caption: "Updates on agents you've published to the store.",
    icon: StorefrontIcon,
    accent: "from-amber-400/15 to-amber-500/0 text-amber-700",
    items: [
      {
        key: "notifyOnAgentApproved",
        title: "Agent approved",
        description: "Your submission is live in the store.",
      },
      {
        key: "notifyOnAgentRejected",
        title: "Agent needs changes",
        description: "Reviewers have feedback before your agent can go live.",
      },
    ],
  },
  {
    id: "balance",
    title: "Credits & balance",
    caption: "Stay ahead of an empty tank.",
    icon: CoinsIcon,
    accent: "from-emerald-400/15 to-emerald-500/0 text-emerald-700",
    items: [
      {
        key: "notifyOnLowBalance",
        title: "Running low",
        description: "Warn me before I run out of credits.",
      },
      {
        key: "notifyOnZeroBalance",
        title: "Empty balance",
        description: "Let me know when my balance hits zero.",
      },
    ],
  },
  {
    id: "summary",
    title: "Summaries",
    caption: "Recap emails to keep tabs on your account at a glance.",
    icon: ChartLineUpIcon,
    accent: "from-sky-400/15 to-sky-500/0 text-sky-700",
    items: [
      {
        key: "notifyOnDailySummary",
        title: "Daily summary",
        description: "A short daily digest of activity.",
      },
      {
        key: "notifyOnWeeklySummary",
        title: "Weekly summary",
        description: "A weekly overview of performance.",
      },
      {
        key: "notifyOnMonthlySummary",
        title: "Monthly summary",
        description: "A comprehensive monthly report.",
      },
    ],
  },
];

export const NOTIFICATIONS_FALLBACK_ICON = BellSimpleRingingIcon;

const PREFERENCE_API_KEYS: Record<NotificationKey, string> = {
  notifyOnAgentRun: "AGENT_RUN",
  notifyOnBlockExecutionFailed: "BLOCK_EXECUTION_FAILED",
  notifyOnContinuousAgentError: "CONTINUOUS_AGENT_ERROR",
  notifyOnAgentApproved: "AGENT_APPROVED",
  notifyOnAgentRejected: "AGENT_REJECTED",
  notifyOnZeroBalance: "ZERO_BALANCE",
  notifyOnLowBalance: "LOW_BALANCE",
  notifyOnDailySummary: "DAILY_SUMMARY",
  notifyOnWeeklySummary: "WEEKLY_SUMMARY",
  notifyOnMonthlySummary: "MONTHLY_SUMMARY",
};

export function preferencesToFlags(
  preferences: NotificationPreference,
): NotificationFlags {
  const raw = preferences.preferences ?? {};
  return Object.fromEntries(
    (Object.keys(PREFERENCE_API_KEYS) as NotificationKey[]).map((key) => [
      key,
      Boolean(raw[PREFERENCE_API_KEYS[key]]),
    ]),
  ) as NotificationFlags;
}

export function flagsToPreferenceMap(
  flags: NotificationFlags,
): Record<string, boolean> {
  return Object.fromEntries(
    (Object.keys(PREFERENCE_API_KEYS) as NotificationKey[]).map((key) => [
      PREFERENCE_API_KEYS[key],
      flags[key],
    ]),
  );
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
  if (initial.timezone !== current.timezone) return true;
  for (const key of Object.keys(initial.notifications) as NotificationKey[]) {
    if (initial.notifications[key] !== current.notifications[key]) return true;
  }
  return false;
}

export function dirtyKinds(
  initial: PreferencesFormState,
  current: PreferencesFormState,
): { timezone: boolean; notifications: boolean } {
  const timezone = initial.timezone !== current.timezone;
  let notifications = false;
  for (const key of Object.keys(initial.notifications) as NotificationKey[]) {
    if (initial.notifications[key] !== current.notifications[key]) {
      notifications = true;
      break;
    }
  }
  return { timezone, notifications };
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

export function formatGmtOffset(timezone: string): string | null {
  try {
    const formatter = new Intl.DateTimeFormat("en-US", {
      timeZone: timezone,
      timeZoneName: "shortOffset",
    });
    const parts = formatter.formatToParts(new Date());
    const offset = parts.find((p) => p.type === "timeZoneName")?.value;
    return offset ?? null;
  } catch {
    return null;
  }
}
