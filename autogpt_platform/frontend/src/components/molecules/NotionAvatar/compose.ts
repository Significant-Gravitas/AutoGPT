import { findNotionColor, type NotionAvatarConfig } from "./helpers";
import {
  NOTION_CATEGORIES,
  VIEWBOX,
  type NotionCategory,
} from "./metadata.generated";
import { NOTION_PARTS } from "./parts.generated";
import { badgeMarkup } from "./statusBadge";
import type { AvatarStatus } from "./status";

// Leave a clear margin around the face while keeping it readable at card size.
export const FRAME_SCALE = 1.2;
export const FRAME_OFFSET_Y = 0;

interface Options {
  size?: number;
  idPrefix?: string;
  /** Omit the tinted disc, for callers that supply their own background. */
  transparent?: boolean;
  /** Draws the status dot on the lower right. "idle" draws nothing. */
  status?: AvatarStatus;
}

export function layerMarkup(
  category: NotionCategory,
  index: number,
  prefix: string,
): string {
  const markup = NOTION_PARTS[category][index] ?? "";
  return markup.replace(/\{\{P\}\}/g, prefix);
}

export function composeNotionAvatar(
  config: NotionAvatarConfig,
  {
    size = VIEWBOX,
    idPrefix = "na",
    transparent = false,
    status = "idle",
  }: Options = {},
): string {
  const color = findNotionColor(config.color);
  const disc = transparent
    ? ""
    : `<circle cx="${VIEWBOX / 2}" cy="${VIEWBOX / 2}" r="${VIEWBOX / 2}" fill="${color.disc}"/>`;
  const layers = NOTION_CATEGORIES.map(
    (category) =>
      `<g>${layerMarkup(category, config.parts[category], idPrefix)}</g>`,
  ).join("");

  const inset = (VIEWBOX / 2) * (1 - FRAME_SCALE);
  const radius = VIEWBOX / 2;
  // The head is drawn larger than the artboard, so the file clips itself to
  // the disc rather than relying on whatever rounds it in CSS.
  const clipId = `${idPrefix}clip`;

  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${VIEWBOX} ${VIEWBOX}" fill="none">`,
    `<defs><clipPath id="${clipId}"><circle cx="${radius}" cy="${radius}" r="${radius}"/></clipPath></defs>`,
    `<g clip-path="url(#${clipId})">`,
    disc,
    `<g transform="translate(${inset} ${inset + FRAME_OFFSET_Y}) scale(${FRAME_SCALE})">${layers}</g>`,
    `</g>`,
    badgeMarkup(status, VIEWBOX),
    `</svg>`,
  ].join("");
}
