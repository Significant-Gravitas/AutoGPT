// Copied from frontend/src/components/molecules/ExpertAvatar/catalog.json
// (`palette` and each identity's `visual_category`); a test keeps them equal.
export const palette = {
  marketing: { label: "Terracotta", hex: "#C47F5C" },
  sales: { label: "Ochre", hex: "#C9A35B" },
  finance: { label: "Sage", hex: "#A5B09A" },
  support: { label: "Muted coral", hex: "#CB9182" },
  operations: { label: "Slate blue", hex: "#98AFC6" },
  research: { label: "Olive", hex: "#AAA77A" },
  content: { label: "Muted teal", hex: "#81AAA6" },
  development: { label: "Charcoal", hex: "#777570" },
  general: { label: "Warm stone", hex: "#B5ADA0" },
  otto: { label: "Mineral lavender", hex: "#B6A4C8" },
} as const;

export type Family = keyof typeof palette;

export const identityFamilies: Record<string, Family> = {
  "expert-maria": "marketing",
  "expert-jules": "marketing",
  "expert-nadia": "research",
  "expert-remy": "marketing",
  "expert-mina": "finance",
  "expert-theo": "finance",
  "expert-quinn": "research",
  "expert-max": "sales",
  "expert-devon": "development",
  "expert-riley": "support",
  "expert-jordan": "sales",
  "expert-frankie": "operations",
  "expert-harper": "operations",
  "expert-vera": "operations",
  "expert-ellis": "operations",
  "expert-robin": "support",
  "expert-anika": "sales",
  "expert-alex": "development",
  "expert-daniel": "finance",
  "expert-sofia": "operations",
  "expert-maya": "marketing",
  "expert-james": "operations",
  "expert-zara": "marketing",
  "expert-sasha": "support",
  "expert-priya": "research",
  "expert-marco": "marketing",
  "expert-noor": "marketing",
  "expert-casey": "development",
  "expert-ines": "operations",
  "expert-omar": "sales",
  "expert-lena": "operations",
  "expert-kai": "support",
  "expert-clip": "marketing",
  "expert-general-01": "general",
  otto: "otto",
};

export const OTTO_TINT: string = palette.otto.hex;

const MANAGED_URL =
  /^(?:https?:\/\/[^/]+)?\/autogpt-characters\/v[\d.]+\/([a-z0-9-]+)\/neutral\/\d+\.(?:webp|png)$/;

function isFamily(value: string): value is Family {
  return value in palette;
}

interface TintArgs {
  avatarUrl?: string | null;
  categories?: readonly string[] | null;
  role?: string | null;
}

/** The expert's area, as the web's `getExpertVisualCategory` picks it. */
export function expertFamily({ avatarUrl, categories, role }: TintArgs): Family {
  if (/head of ai/i.test(role ?? "")) return "otto";
  const id = MANAGED_URL.exec(avatarUrl?.trim() ?? "")?.[1];
  const managed = id ? identityFamilies[id] : undefined;
  if (managed && managed !== "general") return managed;
  const stored = categories
    ?.map((category) => category.toLowerCase())
    .find((category) => isFamily(category) && category !== "otto");
  return (stored as Family | undefined) ?? "general";
}

/** Port of the web's `getExpertTopicHex`. */
export function expertTint(args: TintArgs): string {
  return palette[expertFamily(args)].hex;
}

function channels(hex: string) {
  const value = parseInt(hex.replace("#", ""), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

/** `percent`% of `hex` over `baseHex`, mixed per sRGB channel. */
export function mixHex(hex: string, percent: number, baseHex: string): string {
  const weight = percent / 100;
  const base = channels(baseHex);
  return (
    "#" +
    channels(hex)
      .map((channel, index) =>
        Math.round(channel * weight + base[index] * (1 - weight))
          .toString(16)
          .padStart(2, "0"),
      )
      .join("")
      .toUpperCase()
  );
}
