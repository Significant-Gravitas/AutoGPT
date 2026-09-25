import catalog from "./catalog.json";

export const EXPERT_AVATARS = catalog.avatars;
export const EXPERT_AVATAR_COLORS = catalog.colors;
type BuiltinAvatar = {
  id: string;
  name: string;
  url: string;
  color_id: string;
  previous_url: string;
  previous_urls: readonly string[];
  primary_category: string;
  variants: Partial<Record<string, { url: string; hex: string }>>;
};

export const BUILTIN_EXPERT_AVATARS: readonly BuiltinAvatar[] =
  catalog.identities;
export const DEFAULT_EXPERT_AVATAR_URL = "/experts/clay/v1/content.png";

export function resolveExpertAvatarUrl(url: string | null | undefined): string {
  if (!url) return DEFAULT_EXPERT_AVATAR_URL;
  const legacy: Record<string, string> = catalog.legacy;
  if (legacy[url]) return legacy[url];
  if (url.startsWith("/avatars/notion/") && url.endsWith(".svg")) {
    return DEFAULT_EXPERT_AVATAR_URL;
  }
  return url;
}

export function resolveCategoryAvatarUrl(
  url: string | null | undefined,
  category?: string | null,
): string {
  const resolved = resolveExpertAvatarUrl(url);
  const identity = BUILTIN_EXPERT_AVATARS.find(
    (avatar) =>
      avatar.url === resolved ||
      avatar.previous_urls?.includes(resolved) ||
      Object.values(avatar.variants).some(
        (variant) => variant?.url === resolved,
      ),
  );
  if (!identity) return resolved;
  if (category)
    return identity.variants[category.toLowerCase()]?.url ?? identity.url;
  return identity.previous_urls.includes(resolved) ? identity.url : resolved;
}

import manifest from "../../../../public/autogpt-characters/manifest.json";

export function getManagedAvatar(avatarUrl: string | null, size: number) {
  for (const [assetID, identity] of Object.entries(manifest.identities)) {
    if (
      !Object.values(identity.files).some(
        (file) => `/${file.path.replace(/^public\//, "")}` === avatarUrl,
      )
    )
      continue;
    const pixels = manifest.logicalSizes.find((value) => value >= size) ?? 512;
    return {
      assetID,
      base: `${manifest.baseUrl}/${assetID}/neutral`,
      pixels,
    };
  }
  return null;
}
