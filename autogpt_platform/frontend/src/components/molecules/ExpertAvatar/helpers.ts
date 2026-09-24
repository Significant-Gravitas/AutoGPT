import catalog from "./catalog.json";

export const EXPERT_AVATARS = catalog.avatars;
export const DEFAULT_EXPERT_AVATAR_URL = "/experts/clay/v1/content.png";

export function resolveExpertAvatarUrl(url: string | null | undefined): string {
  if (!url) return DEFAULT_EXPERT_AVATAR_URL;
  const legacy: Record<string, string> = catalog.legacy;
  if (legacy[url]) return legacy[url];
  if (url.startsWith("/avatars/") && url.endsWith(".svg")) {
    return DEFAULT_EXPERT_AVATAR_URL;
  }
  return url;
}
