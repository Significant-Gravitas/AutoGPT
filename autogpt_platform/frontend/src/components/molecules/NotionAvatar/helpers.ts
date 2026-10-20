import {
  NOTION_CATEGORIES,
  NOTION_PART_COUNTS,
  VIEWBOX,
  type NotionCategory,
} from "./metadata.generated";

export { VIEWBOX };

export type NotionParts = Record<NotionCategory, number>;

export interface NotionAvatarConfig {
  parts: NotionParts;
  color: NotionColorId;
}

export type NotionColorId =
  | "rose"
  | "red"
  | "orange"
  | "amber"
  | "yellow"
  | "lime"
  | "green"
  | "emerald"
  | "teal"
  | "cyan"
  | "sky"
  | "blue"
  | "indigo"
  | "violet"
  | "fuchsia";

export interface NotionColorOption {
  id: NotionColorId;
  label: string;
  disc: string;
}

// Keep each disc close to white, with enough hue to match the expert's accent.
export const NOTION_COLORS: NotionColorOption[] = [
  { id: "rose", label: "Rose", disc: "#fff1f2" },
  { id: "red", label: "Red", disc: "#fef2f2" },
  { id: "orange", label: "Orange", disc: "#fff7ed" },
  { id: "amber", label: "Amber", disc: "#fffbeb" },
  { id: "yellow", label: "Yellow", disc: "#fefce8" },
  { id: "lime", label: "Lime", disc: "#f7fee7" },
  { id: "green", label: "Green", disc: "#f0fdf4" },
  { id: "emerald", label: "Emerald", disc: "#ecfdf5" },
  { id: "teal", label: "Teal", disc: "#f0fdfa" },
  { id: "cyan", label: "Cyan", disc: "#ecfeff" },
  { id: "sky", label: "Sky", disc: "#f0f9ff" },
  { id: "blue", label: "Blue", disc: "#eff6ff" },
  { id: "indigo", label: "Indigo", disc: "#eef2ff" },
  { id: "violet", label: "Violet", disc: "#f5f3ff" },
  { id: "fuchsia", label: "Fuchsia", disc: "#fdf4ff" },
];

// Rows the picker offers, in the order they appear: the face itself, then the
// things worn on it.
//
// Every category the seeder can switch on MUST have a row — otherwise a user
// lands on a beard or a pair of blush marks with no way to clear them. That
// rule is enforced by a test against OPTIONAL_CATEGORIES rather than left to
// judgement about which features "read" at small sizes.
export const PICKABLE_CATEGORIES: NotionCategory[] = [
  "hair",
  "eyes",
  "mouth",
  "beard",
  "glasses",
  "accessories",
  "details",
];

export const CATEGORY_LABELS: Record<NotionCategory, string> = {
  face: "Face",
  nose: "Nose",
  mouth: "Mouth",
  eyes: "Eyes",
  eyebrows: "Eyebrows",
  glasses: "Glasses",
  hair: "Hair",
  accessories: "Accessories",
  // Upstream calls these "details"; they are blush, freckles, moles and lines.
  details: "Marks",
  beard: "Beard",
};

export const DEFAULT_NOTION_CONFIG: NotionAvatarConfig = {
  parts: {
    face: 0,
    nose: 0,
    mouth: 0,
    eyes: 0,
    eyebrows: 0,
    glasses: 0,
    hair: 0,
    accessories: 0,
    details: 0,
    beard: 0,
  },
  color: "violet",
};

export function findNotionColor(id: NotionColorId) {
  return NOTION_COLORS.find((color) => color.id === id) ?? NOTION_COLORS[0];
}

function isNotionColorId(value: string): value is NotionColorId {
  return NOTION_COLORS.some((color) => color.id === value);
}

/** Wraps into range rather than rejecting, so a stale or hand-typed URL still
 *  resolves to a real face instead of a 404. */
export function clampPart(category: NotionCategory, index: number): number {
  const count = NOTION_PART_COUNTS[category];
  if (!Number.isFinite(index)) return 0;
  const whole = Math.trunc(index);
  return ((whole % count) + count) % count;
}

export function encodeNotionConfig(config: NotionAvatarConfig): string {
  const slots = NOTION_CATEGORIES.map((category) =>
    clampPart(category, config.parts[category]),
  );
  return `${slots.join("-")}.${config.color}`;
}

export function decodeNotionConfig(value: string): NotionAvatarConfig | null {
  const [slots = "", color = ""] = value.split(".");
  const indices = slots.split("-");
  if (indices.length !== NOTION_CATEGORIES.length) return null;
  if (!indices.every((index) => /^\d+$/.test(index))) return null;
  if (!isNotionColorId(color)) return null;
  const parts = {} as NotionParts;
  NOTION_CATEGORIES.forEach((category, position) => {
    parts[category] = clampPart(category, Number(indices[position]));
  });
  return { parts, color };
}

const NOTION_URL_PATTERN = /^\/avatars\/notion\/([\d-]+\.[a-z]+)\.svg$/;
const NOTION_AVATAR_RENDER_VERSION = 2;

export function notionAvatarUrlFor(config: NotionAvatarConfig): string {
  return `/avatars/notion/${encodeNotionConfig(config)}.svg`;
}

export function notionAvatarImageUrlFor(config: NotionAvatarConfig): string {
  return `${notionAvatarUrlFor(config)}?v=${NOTION_AVATAR_RENDER_VERSION}`;
}

export function parseNotionAvatarUrl(
  url: string | null | undefined,
): NotionAvatarConfig | null {
  const match = url?.match(NOTION_URL_PATTERN);
  return match ? decodeNotionConfig(match[1]) : null;
}

// Iterates by code point (not UTF-16 code unit) so names with characters
// outside the BMP, like emoji, hash the same way as the backend's ord().
export function hashSeed(input: string) {
  let hash = 2166136261;
  for (const character of input) {
    hash ^= character.codePointAt(0) ?? 0;
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

export function seededRandom(seed: number) {
  let state = seed || 1;
  return function next() {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

// Index 0 is "nothing" for these categories. Sampling uniformly would put
// glasses and a beard on almost every face, so each carries the odds of
// drawing nothing at all; the rest are always drawn.
const NONE_ODDS: Partial<Record<NotionCategory, number>> = {
  glasses: 0.55,
  beard: 0.6,
  accessories: 0.7,
  details: 0.65,
  hair: 0.04,
};

/** Categories that can be seeded on, and so must be clearable in the picker. */
export const OPTIONAL_CATEGORIES = Object.keys(NONE_ODDS) as NotionCategory[];

export function weightedPart(
  category: NotionCategory,
  random: () => number,
): number {
  const count = NOTION_PART_COUNTS[category];
  const noneOdds = NONE_ODDS[category];
  if (noneOdds === undefined) return Math.floor(random() * count);
  if (random() < noneOdds) return 0;
  return 1 + Math.floor(random() * (count - 1));
}

export function randomNotionConfig(
  random: () => number = Math.random,
): NotionAvatarConfig {
  const parts = {} as NotionParts;
  NOTION_CATEGORIES.forEach((category) => {
    parts[category] = weightedPart(category, random);
  });
  parts.details = 0;
  return {
    parts,
    color: NOTION_COLORS[Math.floor(random() * NOTION_COLORS.length)].id,
  };
}

export function notionConfigForName(name: string): NotionAvatarConfig {
  return randomNotionConfig(seededRandom(hashSeed(name.toLowerCase())));
}

// One family, one disc, so no two swatches land on the same face.
export function colorForToken(
  token: string | null | undefined,
): NotionColorId | null {
  const family = token?.split("-")[0];
  return family && isNotionColorId(family) ? family : null;
}

// Avatars raised before the Notion art landed are stored as
// "/avatars/<shape>.<color>.<accessory>.svg". They keep resolving to a face —
// seeded from the name, so the same expert always lands on the same one — and
// the legacy route redirects to the Notion URL.
const LEGACY_URL_PATTERN = /^\/avatars\/([a-z]+)\.([a-z]+)\.([a-z]+)\.svg$/;

export function isLegacyAvatarUrl(url: string | null | undefined): boolean {
  return Boolean(url && LEGACY_URL_PATTERN.test(url));
}

/** Seeds from the old triple rather than the expert's name, so the route and
 *  the component resolve a legacy URL to the same face — and two experts who
 *  were raised with different shapes stay different. */
// The palette a legacy URL was written against, mapped onto the family that
// replaced it, so an expert raised under the old names keeps its colour.
const LEGACY_COLORS: Record<string, NotionColorId> = {
  lavender: "violet",
  plum: "fuchsia",
  mint: "emerald",
  coral: "orange",
  butter: "yellow",
  amber: "amber",
  sky: "sky",
  indigo: "indigo",
};

export function notionConfigForLegacyUrl(
  url: string | null | undefined,
): NotionAvatarConfig | null {
  const match = url?.match(LEGACY_URL_PATTERN);
  if (!match) return null;
  const [, shape, color, accessory] = match;
  const seeded = notionConfigForName(`${shape}.${color}.${accessory}`);
  const mapped = LEGACY_COLORS[color];
  return mapped ? { ...seeded, color: mapped } : seeded;
}

export interface ExpertLike {
  name: string | null | undefined;
  avatarUrl?: string | null;
  color?: string | null;
}

/** The config to draw for an expert, or null when their avatar is a real
 *  picture and should be rendered as an image instead. */
export function expertNotionConfig(
  expert: ExpertLike,
): NotionAvatarConfig | null {
  const parsed = parseNotionAvatarUrl(expert.avatarUrl);
  if (parsed) return parsed;

  const legacy = notionConfigForLegacyUrl(expert.avatarUrl);
  if (legacy) return legacy;

  if (expert.avatarUrl) return null;

  const seeded = notionConfigForName(expert.name ?? "");
  const color = colorForToken(expert.color);
  return color ? { ...seeded, color } : seeded;
}
