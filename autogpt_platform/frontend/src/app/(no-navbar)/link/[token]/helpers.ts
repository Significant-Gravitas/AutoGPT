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
