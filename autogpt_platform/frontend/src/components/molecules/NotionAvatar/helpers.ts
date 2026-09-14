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
  | "lavender"
  | "plum"
  | "amber"
  | "sky"
  | "mint"
  | "coral"
  | "indigo"
  | "butter";

export interface NotionColorOption {
  id: NotionColorId;
  label: string;
  role: string;
  disc: string;
  ring: string;
}

export const NOTION_COLORS: NotionColorOption[] = [
  {
    id: "lavender",
    label: "Lavender",
    role: "Otto",
    disc: "#ddd6fe",
    ring: "#a78bfa",
  },
  {
    id: "plum",
    label: "Plum",
    role: "Marketing",
    disc: "#fbcfe8",
    ring: "#f472b6",
  },
  {
    id: "amber",
    label: "Amber",
    role: "Sales",
    disc: "#fde68a",
    ring: "#fbbf24",
  },
  { id: "sky", label: "Sky", role: "Ops", disc: "#bae6fd", ring: "#38bdf8" },
  {
    id: "mint",
    label: "Mint",
    role: "Finance",
    disc: "#a7f3d0",
    ring: "#34d399",
  },
  {
    id: "coral",
    label: "Coral",
    role: "Support",
    disc: "#fed7aa",
    ring: "#fb923c",
  },
  {
    id: "indigo",
    label: "Indigo",
    role: "Research",
    disc: "#c7d2fe",
    ring: "#818cf8",
  },
  {
    id: "butter",
    label: "Butter",
    role: "Content",
    disc: "#fef08a",
    ring: "#facc15",
  },
];

// The features a user can steer in the raise flow: the ones that still read
// once the avatar is scaled down to a chat row.
export const PICKABLE_CATEGORIES: NotionCategory[] = [
  "hair",
  "eyes",
  "mouth",
  "glasses",
  "accessories",
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
  details: "Details",
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
  color: "lavender",
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

export function notionAvatarUrlFor(config: NotionAvatarConfig): string {
  return `/avatars/notion/${encodeNotionConfig(config)}.svg`;
}

export function parseNotionAvatarUrl(
  url: string | null | undefined,
): NotionAvatarConfig | null {
  const match = url?.match(NOTION_URL_PATTERN);
  return match ? decodeNotionConfig(match[1]) : null;
}

export function hashSeed(input: string) {
  let hash = 2166136261;
  for (let index = 0; index < input.length; index += 1) {
    hash ^= input.charCodeAt(index);
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
  return {
    parts,
    color: NOTION_COLORS[Math.floor(random() * NOTION_COLORS.length)].id,
  };
}

export function notionConfigForName(name: string): NotionAvatarConfig {
  return randomNotionConfig(seededRandom(hashSeed(name.toLowerCase())));
}

const TOKEN_COLORS: Record<string, NotionColorId> = {
  rose: "plum",
  red: "coral",
  orange: "coral",
  amber: "amber",
  yellow: "butter",
  lime: "mint",
  green: "mint",
  emerald: "mint",
  teal: "sky",
  cyan: "sky",
  sky: "sky",
  blue: "sky",
  indigo: "indigo",
  violet: "lavender",
  fuchsia: "plum",
};

export function colorForToken(
  token: string | null | undefined,
): NotionColorId | null {
  const family = token?.split("-")[0];
  return family ? (TOKEN_COLORS[family] ?? null) : null;
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
export function notionConfigForLegacyUrl(
  url: string | null | undefined,
): NotionAvatarConfig | null {
  const match = url?.match(LEGACY_URL_PATTERN);
  if (!match) return null;
  const [, shape, color, accessory] = match;
  const seeded = notionConfigForName(`${shape}.${color}.${accessory}`);
  return isNotionColorId(color) ? { ...seeded, color } : seeded;
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
