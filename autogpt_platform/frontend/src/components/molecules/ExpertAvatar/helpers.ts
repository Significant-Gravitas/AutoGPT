import catalog from "./catalog.json";

/** The managed Clay & Rock library: one identity per built-in Expert plus
 *  Otto and the General fallback, each served from a versioned
 *  `/autogpt-characters` path. `catalog.json` is the same file the backend
 *  ships (`avatar_catalog.json`); a test keeps the two copies identical. */
export const MANAGED_IDENTITIES = catalog.identities;
export const EXPERT_PALETTE = catalog.palette;
/** The warm-stone General fallback: where a custom Expert starts and where
 *  an unknown legacy default lands. Never Otto, never someone else's face. */
export const DEFAULT_EXPERT_AVATAR_URL = catalog.default_url;
const MANAGED_LOGICAL_SIZES = [24, 32, 40, 48, 64, 96, 128, 256, 512];

type ManagedIdentity = (typeof catalog.identities)[number];
export type VisualCategory = keyof typeof catalog.palette;

const MANAGED_URL =
  /^(\/autogpt-characters\/v[\d.]+)\/([a-z0-9-]+)\/neutral\/\d+\.(?:webp|png)$/;

/** Stored defaults that no longer ship (Notion SVGs, the retired clay sheets)
 *  resolve to the identity they stood for; uploads, generated images and
 *  current managed URLs pass through unchanged. */
export function resolveExpertAvatarUrl(url: string | null | undefined): string {
  if (!url) return DEFAULT_EXPERT_AVATAR_URL;
  const legacy: Record<string, string> = catalog.legacy;
  if (legacy[url]) return legacy[url];
  if (url.startsWith("/avatars/notion/") && url.endsWith(".svg")) {
    return DEFAULT_EXPERT_AVATAR_URL;
  }
  return url;
}

export function getManagedIdentity(
  url: string | null | undefined,
): ManagedIdentity | null {
  const match = MANAGED_URL.exec(resolveExpertAvatarUrl(url));
  if (!match) return null;
  return (
    MANAGED_IDENTITIES.find(
      (identity) => identity.base_url === match[1] && identity.id === match[2],
    ) ?? null
  );
}

export function getManagedAvatar(
  avatarUrl: string | null | undefined,
  size: number,
) {
  const identity = getManagedIdentity(avatarUrl);
  if (!identity) return null;
  const pixels = MANAGED_LOGICAL_SIZES.find((value) => value >= size) ?? 512;
  return {
    assetID: identity.id,
    base: `${identity.base_url}/${identity.id}/neutral`,
    pixels,
    pngMaxPixels: identity.png_max_pixels,
    identity,
  };
}

export function isVisualCategory(value: string): value is VisualCategory {
  return value in EXPERT_PALETTE;
}

/** The palette family that colors the surfaces around an Expert. A managed
 *  identity carries its own family, so a filter or a category edit never
 *  changes it; a custom appearance takes the first stored category, and no
 *  category at all means the General warm stone. */
export function getExpertVisualCategory(
  avatarUrl: string | null | undefined,
  categories?: readonly string[] | null,
): VisualCategory {
  const identity = getManagedIdentity(avatarUrl);
  if (identity && identity.visual_category !== "general") {
    return identity.visual_category as VisualCategory;
  }
  const stored = categories
    ?.map((category) => category.toLowerCase())
    .find((category) => isVisualCategory(category) && category !== "otto");
  return (stored as VisualCategory | undefined) ?? "general";
}
