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
