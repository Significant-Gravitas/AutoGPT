import type { ExpertAvatarRequest } from "@/app/api/__generated__/models/expertAvatarRequest";
import { ExpertAvatarRequestBase } from "@/app/api/__generated__/models/expertAvatarRequestBase";
import type { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import { ExpertAvatarRequestExpression } from "@/app/api/__generated__/models/expertAvatarRequestExpression";
import { ExpertAvatarRequestInlay } from "@/app/api/__generated__/models/expertAvatarRequestInlay";
import { ExpertAvatarRequestShape } from "@/app/api/__generated__/models/expertAvatarRequestShape";
import { ExpertAvatarRequestTilt } from "@/app/api/__generated__/models/expertAvatarRequestTilt";
import {
  DEFAULT_EXPERT_AVATAR_URL,
  MANAGED_IDENTITIES,
} from "../ExpertAvatar/helpers";

export const ACCEPTED_AVATAR_TYPES = "image/png,image/jpeg,image/webp";

export function defaultAvatarUrl(
  category: ExpertAvatarRequestCategory,
  name: string,
) {
  const family = MANAGED_IDENTITIES.filter(
    (identity) => identity.visual_category === category,
  );
  const candidates = family.length
    ? family
    : MANAGED_IDENTITIES.filter((identity) =>
        identity.categories.includes(category),
      );
  let hash = 0;
  for (const character of name.trim().toLowerCase()) {
    hash = (hash * 31 + character.charCodeAt(0)) >>> 0;
  }
  return candidates[hash % candidates.length]?.url ?? DEFAULT_EXPERT_AVATAR_URL;
}

/** Everything except the category is left to chance, so regenerating shuffles
 *  the figure while the hue stays the one the category answered for. */
export function randomAvatarRequest(
  category: ExpertAvatarRequestCategory,
): ExpertAvatarRequest {
  return {
    category,
    shape: pick(Object.values(ExpertAvatarRequestShape)),
    base: pick(Object.values(ExpertAvatarRequestBase)),
    tilt: pick(Object.values(ExpertAvatarRequestTilt)),
    inlay: pick(Object.values(ExpertAvatarRequestInlay)),
    expression: pick(Object.values(ExpertAvatarRequestExpression)),
  };
}

function pick<T>(values: readonly T[]): T {
  return values[Math.floor(Math.random() * values.length)];
}
