import {
  EXPERT_PALETTE,
  getExpertVisualCategory,
  isVisualCategory,
} from "./helpers";

/** Otto's reserved lavender, the one color no category can take. */
export const AUTOPILOT_HEX = EXPERT_PALETTE.otto.hex;

/** The design system's material anchor for a category; `undefined` for a
 *  value that is not a palette family. */
export function getCategoryHex(
  category: string | null | undefined,
): string | undefined {
  const key = category?.toLowerCase();
  return key && isVisualCategory(key) ? EXPERT_PALETTE[key].hex : undefined;
}

interface TopicArgs {
  avatarUrl?: string | null;
  categories?: readonly string[] | null;
  /** Only Otto's role picks a color; a specialist's role never does. */
  role?: string | null;
}

/** The hex that tints an Expert's card band, cover and page header. */
export function getExpertTopicHex({
  avatarUrl,
  categories,
  role,
}: TopicArgs): string {
  if (/head of ai/i.test(role ?? "")) return AUTOPILOT_HEX;
  return EXPERT_PALETTE[getExpertVisualCategory(avatarUrl, categories)].hex;
}

export function expertPastel(hex: string): string {
  return `color-mix(in srgb, ${hex} 24%, white)`;
}
