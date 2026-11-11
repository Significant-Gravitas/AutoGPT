// Relative imports keep this module free of the auth client, so its test loads
// without native modules.
import { PLAN_REQUIRED_MESSAGE } from "../chat/chatErrors";
import type { CreateExpertBody } from "../../lib/api/experts";
import { palette } from "../../lib/expertPalette";

import { managedAvatars } from "./catalog";

export type Family = (typeof managedAvatars)[number]["family"];

// The web's raise flow (`raise/components/CategoryStep/helpers.ts`): picking
// an area there sets the expert's `color` token and `role` together.
const FAMILY_DETAILS: Record<Family, { color: string; role: string }> = {
  marketing: { color: "orange-300", role: "Marketer" },
  sales: { color: "amber-300", role: "Sales" },
  finance: { color: "green-300", role: "Analyst" },
  support: { color: "rose-300", role: "Support" },
  operations: { color: "sky-300", role: "Operations" },
  research: { color: "lime-300", role: "Researcher" },
  development: { color: "blue-300", role: "Developer" },
  general: { color: "amber-300", role: "Assistant" },
};

export const NAME_MAX_LENGTH = 100;
export const JOB_TITLE_MAX_LENGTH = 100;

/** Where a new expert starts, as on the web. */
export const DEFAULT_FAMILY: Family = "general";

/**
 * One swatch per family a new expert can wear, in the palette's order. An
 * expert's colour comes from its avatar, so a family with no avatar of its
 * own (content, and Otto's) is left out.
 */
export const colourFamilies = (Object.keys(palette) as (keyof typeof palette)[])
  .filter((key): key is Family => managedAvatars.some((avatar) => avatar.family === key))
  .map((family) => ({ family, label: palette[family].label, hex: palette[family].hex }));

/** Port of the web's `defaultAvatarUrl`: the family's avatar the name hashes to. */
export function defaultAvatarPath(family: Family, name: string) {
  const candidates = managedAvatars.filter((avatar) => avatar.family === family);
  let hash = 0;
  for (const character of name.trim().toLowerCase()) {
    hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  }
  return candidates[hash % candidates.length].url;
}

export function canCreate(name: string) {
  return name.trim().length > 0;
}

interface Draft {
  name: string;
  jobTitle: string;
  family: Family;
}

export function createExpertBody({ name, jobTitle, family }: Draft): CreateExpertBody {
  const trimmedName = name.trim();
  return {
    name: trimmedName,
    role: FAMILY_DETAILS[family].role,
    job_title: jobTitle.trim() || null,
    avatar_url: defaultAvatarPath(family, trimmedName),
    color: FAMILY_DETAILS[family].color,
  };
}

interface Refusal {
  status?: unknown;
  code?: unknown;
  limit?: unknown;
}

/** The sentence for a create that failed for any reason but a dead session. */
export function createErrorMessage(error: unknown) {
  const { status, code, limit } =
    typeof error === "object" && error !== null ? (error as Refusal) : {};
  if (status === 402) return PLAN_REQUIRED_MESSAGE;
  if (status === 409 && code === "active_expert_limit") {
    const size = typeof limit === "number" ? ` (${limit} experts)` : "";
    return `Your team is full${size}. Remove one on the web to add another.`;
  }
  if (status === 409 && code === "raised_expert_lifetime_limit") {
    return "You've created the most experts your plan allows.";
  }
  return "Couldn't create that expert. Try again.";
}
