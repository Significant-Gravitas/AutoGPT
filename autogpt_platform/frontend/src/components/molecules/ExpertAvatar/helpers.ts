import catalog from "./catalog.json";

export const EXPERT_AVATARS = catalog.avatars;
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
