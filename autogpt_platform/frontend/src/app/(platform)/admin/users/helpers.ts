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

export function toAdminUser(record: AuthUserRecord): AdminUser {
  return {
    id: record.id,
    email: record.email,
    name: record.name,
    image: record.image ?? null,
    role: record.role ?? null,
    emailVerified: record.emailVerified,
    createdAt: new Date(record.createdAt),
    banned: Boolean(record.banned),
    banReason: record.banReason ?? null,
    banExpires: record.banExpires ? new Date(record.banExpires) : null,
  };
}

export function getBanDurationSeconds(duration: BanDuration) {
  return BAN_DURATIONS.find((option) => option.value === duration)?.seconds;
}

export function getAdminUsersHref(user?: string | null) {
  const value = user?.trim();
  if (!value) return "/admin/users";
  return `/admin/users?${new URLSearchParams({ [USER_QUERY_PARAM]: value })}`;
}

// The spending search matches the user ID by substring; a full UUID holds no
// LIKE wildcards, so this only ever matches that user.
export function getUserSpendingHref(userId: string) {
  return `/admin/spending?${new URLSearchParams({ search: userId })}`;
}

export function getUserPlatformCostsHref(userId: string) {
  return `/admin/platform-costs?${new URLSearchParams({ user_id: userId })}`;
}
