import type { RelatedAccountPaymentStatus } from "@/app/api/__generated__/models/relatedAccountPaymentStatus";

export const USER_QUERY_PARAM = "user";

export interface LookupQuery {
  kind: "id" | "email";
  value: string;
}

export interface AdminUser {
  id: string;
  email: string;
  name: string;
  image: string | null;
  role: string | null;
  emailVerified: boolean;
  createdAt: Date;
  banned: boolean;
  banReason: string | null;
  banExpires: Date | null;
}

// Better Auth's admin endpoints return the auth identity row with dates
// already revived or as ISO strings, depending on the client's parser.
export interface AuthUserRecord {
  id: string;
  email: string;
  name: string;
  image?: string | null;
  role?: string | null;
  emailVerified: boolean;
  createdAt: Date | string;
  banned?: boolean | null;
  banReason?: string | null;
  banExpires?: Date | string | null;
}

export const BAN_DURATIONS = [
  { value: "permanent", label: "Permanent", seconds: undefined },
  { value: "1h", label: "1 hour", seconds: 60 * 60 },
  { value: "1d", label: "1 day", seconds: 60 * 60 * 24 },
  { value: "7d", label: "7 days", seconds: 60 * 60 * 24 * 7 },
  { value: "30d", label: "30 days", seconds: 60 * 60 * 24 * 30 },
] as const;

export type BanDuration = (typeof BAN_DURATIONS)[number]["value"];

export function parseLookupQuery(
  raw: string | null | undefined,
): LookupQuery | null {
  const value = raw?.trim();
  if (!value) return null;
  // Better Auth stores emails lowercased, so this makes the exact match
  // case-insensitive without a LIKE.
  if (value.includes("@")) return { kind: "email", value: value.toLowerCase() };
  return { kind: "id", value };
}

export function toAdminUser(
  record: AuthUserRecord,
  now = new Date(),
): AdminUser {
  const banExpires = record.banExpires ? new Date(record.banExpires) : null;
  // Better Auth clears a lapsed ban only at the user's next sign-in.
  const banned = Boolean(record.banned) && !(banExpires && banExpires <= now);
  return {
    id: record.id,
    email: record.email,
    name: record.name,
    image: record.image ?? null,
    role: record.role ?? null,
    emailVerified: record.emailVerified,
    createdAt: new Date(record.createdAt),
    banned,
    banReason: banned ? (record.banReason ?? null) : null,
    banExpires: banned ? banExpires : null,
  };
}

export function getBanDurationSeconds(duration: BanDuration) {
  return BAN_DURATIONS.find((option) => option.value === duration)?.seconds;
}

// A repeated query param (`?user=a&user=b`) reaches a page as an array.
export function getAdminUsersHref(user?: string | string[] | null) {
  const value = (Array.isArray(user) ? user[0] : user)?.trim();
  if (!value) return "/admin/users";
  return `/admin/users?${new URLSearchParams({ [USER_QUERY_PARAM]: value })}`;
}

// The spending search is a substring match on user ID, email and name. A full
// UUID holds no LIKE wildcards, so searching one only ever matches that user.
export function getSpendingSearchHref(search: string) {
  return `/admin/spending?${new URLSearchParams({ search })}`;
}

export function getUserPlatformCostsHref(userId: string) {
  return `/admin/platform-costs?${new URLSearchParams({ user_id: userId })}`;
}

// Pinned to en-GB so dates read "12 Mar 2026" and times 24h whatever the
// admin's browser locale.
const DATE_FORMAT = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
});
const DATE_TIME_FORMAT = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
  hour12: false,
});
const MONTH_FORMAT = new Intl.DateTimeFormat("en-GB", {
  month: "short",
  year: "numeric",
});
const RELATIVE_FORMAT = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

const HOUR_MS = 60 * 60 * 1000;
const DAY_MS = 24 * HOUR_MS;

export function formatDate(date: Date) {
  return DATE_FORMAT.format(date);
}

export function formatDateTime(date: Date) {
  return DATE_TIME_FORMAT.format(date);
}

export function formatMonth(date: Date) {
  return MONTH_FORMAT.format(date);
}

export function formatTimeUntil(date: Date, now = new Date()) {
  const ms = date.getTime() - now.getTime();
  if (Math.abs(ms) < DAY_MS) {
    return RELATIVE_FORMAT.format(Math.round(ms / HOUR_MS), "hour");
  }
  return RELATIVE_FORMAT.format(Math.round(ms / DAY_MS), "day");
}

export function getBanEnd(duration: BanDuration, now = new Date()) {
  const seconds = getBanDurationSeconds(duration);
  return seconds === undefined
    ? null
    : new Date(now.getTime() + seconds * 1000);
}

export function getDisplayName(user: Pick<AdminUser, "name" | "email">) {
  return user.name.trim() || user.email;
}

const PLAN_NAMES: Record<string, string> = {
  NO_TIER: "No plan",
  TRIAL: "Trial",
  BASIC: "Basic",
  PRO: "Pro",
  MAX: "Max",
  BUSINESS: "Business",
  ENTERPRISE: "Enterprise",
};

export function getPlanName(tier: string) {
  return PLAN_NAMES[tier] ?? tier;
}

// Orval types a mutation's error as the response body, but the mutator throws
// an ApiError carrying the server's message.
export function getErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : "Something went wrong.";
}

export const TAB_QUERY_PARAM = "tab";
export const BLOCKS_TAB = "blocks";

export function pluralize(count: number, singular: string, plural?: string) {
  const noun = count === 1 ? singular : (plural ?? `${singular}s`);
  return `${count.toLocaleString("en-GB")} ${noun}`;
}

const BROWSERS: Array<[RegExp, string]> = [
  [/Edg\/(\d+)/, "Edge"],
  [/OPR\/(\d+)/, "Opera"],
  [/Firefox\/(\d+)/, "Firefox"],
  [/Chrome\/(\d+)/, "Chrome"],
  [/Version\/[\d.]+.*Safari/, "Safari"],
];
const SYSTEMS: Array<[RegExp, string]> = [
  [/iPhone/, "iPhone"],
  [/iPad/, "iPad"],
  [/Android/, "Android"],
  [/Windows/, "Windows"],
  [/Mac OS X|Macintosh/, "macOS"],
  [/Linux/, "Linux"],
];

// "Chrome 141 · macOS"; anything unrecognised as its first 40 characters.
export function summariseUserAgent(userAgent: string | null | undefined) {
  if (!userAgent) return "Unknown device";
  const browser = BROWSERS.find(([pattern]) => pattern.test(userAgent));
  const system = SYSTEMS.find(([pattern]) => pattern.test(userAgent));
  if (!browser && !system) return userAgent.slice(0, 40);
  const version = browser?.[0].exec(userAgent)?.[1];
  const browserLabel = browser
    ? [browser[1], version].filter(Boolean).join(" ")
    : null;
  return [browserLabel, system?.[1]].filter(Boolean).join(" · ");
}

// The backend stores an IPv6 client by its /64 written out in full
// ("2001:0db8:0000:…"); this is the form a person reads ("2001:db8::/64").
export function formatBlockedAddress(value: string) {
  if (!value.includes(":")) return value;
  const groups = value.split(":").map((group) => group.replace(/^0+(?=.)/, ""));
  let bestStart = -1;
  let bestLength = 1;
  for (let i = 0; i < groups.length; ) {
    if (groups[i] !== "0") {
      i++;
      continue;
    }
    let end = i;
    while (end < groups.length && groups[end] === "0") end++;
    if (end - i > bestLength) {
      bestStart = i;
      bestLength = end - i;
    }
    i = end;
  }
  const compressed =
    bestStart === -1
      ? groups.join(":")
      : `${groups.slice(0, bestStart).join(":")}::${groups
          .slice(bestStart + bestLength)
          .join(":")}`;
  return `${compressed}/64`;
}

export function formatEnds(expiresAt: Date | null, now = new Date()) {
  if (!expiresAt) return "Permanent";
  return `${formatDateTime(expiresAt)} (${formatTimeUntil(expiresAt, now)})`;
}

const PAYMENT_STATUS_LABELS: Record<RelatedAccountPaymentStatus, string> = {
  trialing: "Trialing",
  active: "Active",
  past_due: "Past due",
  canceled: "Canceled",
  granted: "Granted, not billed",
  no_subscription: "No subscription",
};

// "Pro · Active"; an account with neither reads "No plan".
export function getSubscriptionLabel(
  plan: string | null | undefined,
  status: RelatedAccountPaymentStatus,
) {
  if (status === "no_subscription" && (!plan || plan === "NO_TIER")) {
    return "No plan";
  }
  if (status === "trialing" && plan === "TRIAL") return "Trial";
  return `${getPlanName(plan ?? "NO_TIER")} · ${PAYMENT_STATUS_LABELS[status]}`;
}
