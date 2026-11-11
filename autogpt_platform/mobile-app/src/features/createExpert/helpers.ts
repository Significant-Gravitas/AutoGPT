// Relative imports keep this module free of the auth client, so its test loads
// without native modules.
import { PLAN_REQUIRED_MESSAGE } from "../chat/chatErrors";
import type { CreateExpertBody } from "../../lib/api/experts";
import { palette } from "../../lib/expertPalette";

import { managedAvatars } from "./catalog";

export type Family = (typeof managedAvatars)[number]["family"];

// The web's raise flow (`raise/components/CategoryStep/helpers.ts`): picking
// an area there sets the expert's `color` token and `role` together.
const FAMILY_DETAILS: Record<Family, { label: string; color: string; role: string }> = {
  marketing: { label: "Marketing", color: "orange-300", role: "Marketer" },
  sales: { label: "Sales", color: "amber-300", role: "Sales" },
  finance: { label: "Finance", color: "green-300", role: "Analyst" },
  support: { label: "Support", color: "rose-300", role: "Support" },
  operations: { label: "Operations", color: "sky-300", role: "Operations" },
  research: { label: "Research", color: "lime-300", role: "Researcher" },
  development: { label: "Development", color: "blue-300", role: "Developer" },
  general: { label: "General", color: "amber-300", role: "Assistant" },
};

export const NAME_MAX_LENGTH = 100;
export const JOB_TITLE_MAX_LENGTH = 100;

/** Where a new expert starts, as on the web. */
export const DEFAULT_FAMILY: Family = "general";

/**
 * The areas a new expert can work in, General first, then the palette's order.
 * The area sets the expert's role as well as its colour, so it is offered by
 * name, as on the web. An area with no avatar of its own (content) is left out.
 */
export const areaOptions = (Object.keys(palette) as (keyof typeof palette)[])
  .filter((key): key is Family => managedAvatars.some((avatar) => avatar.family === key))
  .sort((a, b) => Number(b === DEFAULT_FAMILY) - Number(a === DEFAULT_FAMILY))
  .map((family) => ({
    family,
    label: FAMILY_DETAILS[family].label,
    hex: palette[family].hex,
  }));

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
