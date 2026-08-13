import { LinkType } from "@/app/api/__generated__/models/linkType";

export const PLATFORM_NAMES: Record<string, string> = {
  DISCORD: "Discord",
  TELEGRAM: "Telegram",
  SLACK: "Slack",
  TEAMS: "Teams",
  WHATSAPP: "WhatsApp",
  GITHUB: "GitHub",
  LINEAR: "Linear",
};

// Matches backend's Path validation on /tokens/{token}/... — URL-safe base64
// characters, bounded length. Keeps malformed params out of proxy fetches.
export const TOKEN_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;

export function getPlatformDisplayName(raw: string | null | undefined): string {
  if (!raw) return "chat platform";
  return PLATFORM_NAMES[raw.toUpperCase()] ?? raw;
}

export function getLoginRedirect(token: string | null): string {
  const next = token ? `/link/${token}` : "/";
  return `/login?next=${encodeURIComponent(next)}`;
}

export function isUserLink(linkType: LinkType | undefined): boolean {
  return linkType === LinkType.USER;
}

const TELEGRAM_AUTH_KEYS = [
  "id",
  "first_name",
  "last_name",
  "username",
  "photo_url",
  "auth_date",
  "hash",
] as const;

export function getTelegramAuth(
  searchParams: URLSearchParams,
): Record<string, string> | null {
  if (!searchParams.get("id") || !searchParams.get("hash")) return null;
  const auth: Record<string, string> = {};
  for (const key of TELEGRAM_AUTH_KEYS) {
    const value = searchParams.get(key);
    if (value !== null) auth[key] = value;
  }
  return auth;
}
