export type ShapeId = "round" | "dome" | "squircle" | "wide" | "bean";
export type ColorId =
  | "lavender"
  | "plum"
  | "amber"
  | "sky"
  | "mint"
  | "coral"
  | "indigo"
  | "butter";
export type AccessoryId =
  | "none"
  | "glasses"
  | "headset"
  | "star"
  | "bow"
  | "badge"
  | "crown"
  | "propeller"
  | "ears"
  | "flower"
  | "bowtie"
  | "headband";
export type AvatarStatus =
  | "idle"
  | "thinking"
  | "working"
  | "waiting"
  | "done"
  | "failed"
  | "sleeping";

export interface AvatarConfig {
  shape: ShapeId;
  color: ColorId;
  accessory: AccessoryId;
}

export interface ShapeAnchors {
  cx: number;
  eyeY: number;
  eyeGap: number;
  top: number;
  bottom: number;
  width: number;
}

export interface ShapeOption {
  id: ShapeId;
  label: string;
  hint: string;
  path: string;
  anchors: ShapeAnchors;
}

export interface ColorOption {
  id: ColorId;
  label: string;
  role: string;
  light: string;
  body: string;
  mid: string;
  deep: string;
}

export interface AccessoryOption {
  id: AccessoryId;
  label: string;
  hint: string;
}

export interface StatusOption {
  id: AvatarStatus;
  label: string;
  hint: string;
}

export const VIEWBOX = 120;
export const INK = "#26262E";

export const SHAPES: ShapeOption[] = [
  {
    id: "round",
    label: "Round",
    hint: "round, friendly",
    path: "M60,22 A42,42 0 1 1 59.9,22 Z",
    anchors: { cx: 60, eyeY: 62, eyeGap: 24, top: 22, bottom: 106, width: 84 },
  },
  {
    id: "dome",
    label: "Dome",
    hint: "closest to Otto",
    path: "M24,102 L24,64 A36,36 0 0 1 96,64 L96,102 Q96,108 90,108 L30,108 Q24,108 24,102 Z",
    anchors: { cx: 60, eyeY: 64, eyeGap: 24, top: 28, bottom: 108, width: 72 },
  },
  {
    id: "squircle",
    label: "Squircle",
    hint: "steady, square",
    path: "M60,24 C92,24 100,32 100,64 C100,96 92,104 60,104 C28,104 20,96 20,64 C20,32 28,24 60,24 Z",
    anchors: { cx: 60, eyeY: 62, eyeGap: 26, top: 24, bottom: 104, width: 80 },
  },
  {
    id: "wide",
    label: "Wide",
    hint: "low and wide",
    path: "M60,38 C92,38 108,52 108,72 C108,92 92,106 60,106 C28,106 12,92 12,72 C12,52 28,38 60,38 Z",
    anchors: { cx: 60, eyeY: 68, eyeGap: 28, top: 38, bottom: 106, width: 96 },
  },
  {
    id: "bean",
    label: "Bean",
    hint: "lively, off-centre",
    path: "M34,40 C36,22 56,16 72,22 C92,30 102,50 98,72 C94,94 76,108 54,104 C32,100 20,80 26,60 C28,52 33,48 34,40 Z",
    anchors: { cx: 62, eyeY: 58, eyeGap: 24, top: 18, bottom: 106, width: 76 },
  },
];

export const COLORS: ColorOption[] = [
  {
    id: "lavender",
    label: "Lavender",
    role: "AutoPilot",
    light: "#c4b5fd",
    body: "#a78bfa",
    mid: "#8b5cf6",
    deep: "#6d28d9",
  },
  {
    id: "plum",
    label: "Plum",
    role: "Marketing",
    light: "#f9a8d4",
    body: "#f472b6",
    mid: "#ec4899",
    deep: "#be185d",
  },
  {
    id: "amber",
    label: "Amber",
    role: "Sales",
    light: "#fcd34d",
    body: "#fbbf24",
    mid: "#f59e0b",
    deep: "#b45309",
  },
  {
    id: "sky",
    label: "Sky",
    role: "Ops",
    light: "#7dd3fc",
    body: "#38bdf8",
    mid: "#0ea5e9",
    deep: "#0369a1",
  },
  {
    id: "mint",
    label: "Mint",
    role: "Finance",
    light: "#6ee7b7",
    body: "#34d399",
    mid: "#10b981",
    deep: "#047857",
  },
  {
    id: "coral",
    label: "Coral",
    role: "Support",
    light: "#fdba74",
    body: "#fb923c",
    mid: "#f97316",
    deep: "#c2410c",
  },
  {
    id: "indigo",
    label: "Indigo",
    role: "Research",
    light: "#a5b4fc",
    body: "#818cf8",
    mid: "#6366f1",
    deep: "#4338ca",
  },
  {
    id: "butter",
    label: "Butter",
    role: "Content",
    light: "#fde047",
    body: "#facc15",
    mid: "#eab308",
    deep: "#a16207",
  },
];

export const ACCESSORIES: AccessoryOption[] = [
  { id: "none", label: "None", hint: "plain" },
  { id: "glasses", label: "Glasses", hint: "research, review" },
  { id: "headset", label: "Headset", hint: "support, ops" },
  { id: "star", label: "Star pin", hint: "a raised favourite" },
  { id: "bow", label: "Bow", hint: "marketing, events" },
  { id: "badge", label: "Badge", hint: "finance, admin" },
  { id: "crown", label: "Crown", hint: "the lead of a pod" },
  { id: "propeller", label: "Propeller", hint: "playful, experiments" },
  { id: "ears", label: "Cat ears", hint: "curious, alert" },
  { id: "flower", label: "Flower", hint: "people, community" },
  { id: "bowtie", label: "Bow tie", hint: "formal, legal" },
  { id: "headband", label: "Headband", hint: "focus, sprints" },
];

export const STATUSES: StatusOption[] = [
  { id: "idle", label: "Idle", hint: "calm, slightly curious" },
  { id: "thinking", label: "Thinking", hint: "brows wave, eyes up" },
  { id: "working", label: "Working", hint: "kicks into gear" },
  { id: "waiting", label: "Needs you", hint: "blocked, asking" },
  { id: "done", label: "Done", hint: "settles, satisfied" },
  { id: "failed", label: "Failed", hint: "falls apart for a beat" },
  { id: "sleeping", label: "Paused", hint: "schedules off, dozing" },
];

export const AUTOPILOT_AVATAR: AvatarConfig = {
  shape: "dome",
  color: "lavender",
  accessory: "none",
};

export const DEFAULT_CONFIG: AvatarConfig = {
  shape: "round",
  color: "lavender",
  accessory: "none",
};

export function findShape(id: ShapeId) {
  return SHAPES.find((shape) => shape.id === id) ?? SHAPES[0];
}

export function findColor(id: ColorId) {
  return COLORS.find((color) => color.id === id) ?? COLORS[0];
}

function isShapeId(value: string): value is ShapeId {
  return SHAPES.some((shape) => shape.id === value);
}

function isColorId(value: string): value is ColorId {
  return COLORS.some((color) => color.id === value);
}

function isAccessoryId(value: string): value is AccessoryId {
  return ACCESSORIES.some((accessory) => accessory.id === value);
}

export function isAvatarStatus(value: string): value is AvatarStatus {
  return STATUSES.some((status) => status.id === value);
}

export function encodeConfig(config: AvatarConfig) {
  return `${config.shape}.${config.color}.${config.accessory}`;
}

export function decodeConfig(value: string | null): AvatarConfig {
  if (!value) return DEFAULT_CONFIG;
  const [shape = "", color = "", accessory = ""] = value.split(".");
  return {
    shape: isShapeId(shape) ? shape : DEFAULT_CONFIG.shape,
    color: isColorId(color) ? color : DEFAULT_CONFIG.color,
    accessory: isAccessoryId(accessory) ? accessory : DEFAULT_CONFIG.accessory,
  };
}

function pick<T>(items: readonly T[], random: () => number) {
  return items[Math.floor(random() * items.length)];
}

export function randomConfig(random: () => number = Math.random): AvatarConfig {
  return {
    shape: pick(SHAPES, random).id,
    color: pick(COLORS, random).id,
    accessory: pick(ACCESSORIES, random).id,
  };
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

export function configForName(name: string): AvatarConfig {
  return randomConfig(seededRandom(hashSeed(name.toLowerCase())));
}

const AVATAR_URL_PATTERN = /^\/avatars\/([a-z]+)\.([a-z]+)\.([a-z]+)\.svg$/;

export function avatarUrlFor(config: AvatarConfig) {
  return `/avatars/${encodeConfig(config)}.svg`;
}

export function parseAvatarUrl(
  url: string | null | undefined,
): AvatarConfig | null {
  const match = url?.match(AVATAR_URL_PATTERN);
  if (!match) return null;
  const [, shape, color, accessory] = match;
  if (!isShapeId(shape) || !isColorId(color) || !isAccessoryId(accessory))
    return null;
  return { shape, color, accessory };
}

const TOKEN_COLORS: Record<string, ColorId> = {
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
): ColorId | null {
  const family = token?.split("-")[0];
  return family ? (TOKEN_COLORS[family] ?? null) : null;
}

export interface ExpertLike {
  name: string | null | undefined;
  avatarUrl?: string | null;
  color?: string | null;
}

export function expertAvatarConfig(expert: ExpertLike): AvatarConfig {
  const parsed = parseAvatarUrl(expert.avatarUrl);
  if (parsed) return parsed;
  const seeded = configForName(expert.name ?? "");
  const color = colorForToken(expert.color);
  return color ? { ...seeded, color } : seeded;
}

export function isUploadedAvatar(url: string | null | undefined) {
  return Boolean(url) && !parseAvatarUrl(url);
}
