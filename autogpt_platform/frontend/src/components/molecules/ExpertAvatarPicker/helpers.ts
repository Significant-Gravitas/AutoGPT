import type { ExpertAvatarRequest } from "@/app/api/__generated__/models/expertAvatarRequest";
import { ExpertAvatarRequestAccentCount } from "@/app/api/__generated__/models/expertAvatarRequestAccentCount";
import { ExpertAvatarRequestAccentPlacement } from "@/app/api/__generated__/models/expertAvatarRequestAccentPlacement";
import { ExpertAvatarRequestBase } from "@/app/api/__generated__/models/expertAvatarRequestBase";
import type { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import { ExpertAvatarRequestExpression } from "@/app/api/__generated__/models/expertAvatarRequestExpression";
import { ExpertAvatarRequestInlay } from "@/app/api/__generated__/models/expertAvatarRequestInlay";
import { ExpertAvatarRequestShape } from "@/app/api/__generated__/models/expertAvatarRequestShape";
import { ExpertAvatarRequestTilt } from "@/app/api/__generated__/models/expertAvatarRequestTilt";

export const ACCEPTED_AVATAR_TYPES = "image/png,image/jpeg,image/webp";
export const MAX_AVATAR_BYTES = 5 * 1024 * 1024;

const SHADES = ["standard", "light", "dark"] as const;

/** Everything except the category is left to chance, so regenerating shuffles
 *  the figure while the hue stays the one the category answered for. */
export function randomAvatarRequest(
  category: ExpertAvatarRequestCategory,
): ExpertAvatarRequest {
  return {
    category,
    shade: pick(SHADES),
    shape: pick(Object.values(ExpertAvatarRequestShape)),
    base: pick(Object.values(ExpertAvatarRequestBase)),
    tilt: pick(Object.values(ExpertAvatarRequestTilt)),
    inlay: pick(Object.values(ExpertAvatarRequestInlay)),
    accent_placement: pick(Object.values(ExpertAvatarRequestAccentPlacement)),
    accent_count: pick(Object.values(ExpertAvatarRequestAccentCount)),
    expression: pick(Object.values(ExpertAvatarRequestExpression)),
  };
}

function pick<T>(values: readonly T[]): T {
  return values[Math.floor(Math.random() * values.length)];
}
