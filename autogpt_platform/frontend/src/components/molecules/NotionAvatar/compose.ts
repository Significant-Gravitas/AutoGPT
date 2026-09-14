import { findNotionColor, type NotionAvatarConfig } from "./helpers";
import {
  NOTION_CATEGORIES,
  VIEWBOX,
  type NotionCategory,
} from "./metadata.generated";
import { NOTION_PARTS } from "./parts.generated";

// The artwork is drawn to the edge of its 1080 box, so a circular crop shears
// off long hair. Scaling it down inside the disc keeps every hairstyle whole,
// and the nudge downward stops the crown from crowding the top edge.
export const FRAME_SCALE = 0.8;
export const FRAME_OFFSET_Y = 20;

interface Options {
  size?: number;
  idPrefix?: string;
  /** Omit the tinted disc, for callers that supply their own background. */
  transparent?: boolean;
}

// A white halo dilated out of the line art, so the drawing stays legible on a
// saturated disc. Carried over from the upstream editor.
export function haloFilter(prefix: string, innerOnly = false) {
  const filter = `<filter id="${prefix}halo" x="-20%" y="-20%" width="140%" height="140%" filterUnits="objectBoundingBox" primitiveUnits="userSpaceOnUse" color-interpolation-filters="linearRGB"><feMorphology operator="dilate" radius="20 20" in="SourceAlpha" result="morphology"/><feFlood flood-color="#ffffff" flood-opacity="1" result="flood"/><feComposite in="flood" in2="morphology" operator="in" result="composite"/><feMerge result="merge"><feMergeNode in="composite"/><feMergeNode in="SourceGraphic"/></feMerge></filter>`;
  return innerOnly ? filter : `<defs>${filter}</defs>`;
}

export function layerMarkup(
  category: NotionCategory,
  index: number,
  prefix: string,
): string {
  const markup = NOTION_PARTS[category][index] ?? "";
  return markup.replace(/\{\{P\}\}/g, prefix);
}

/** The face layer is the only one that needs a fill: left transparent, the
 *  tinted disc would show through the skin. */
export function layerFill(category: NotionCategory): string | undefined {
  return category === "face" ? "#ffffff" : undefined;
}

export function composeNotionAvatar(
  config: NotionAvatarConfig,
  { size = VIEWBOX, idPrefix = "na", transparent = false }: Options = {},
): string {
  const color = findNotionColor(config.color);
  const disc = transparent
    ? ""
    : `<circle cx="${VIEWBOX / 2}" cy="${VIEWBOX / 2}" r="${VIEWBOX / 2}" fill="${color.disc}"/>`;
  const layers = NOTION_CATEGORIES.map((category) => {
    const fill = layerFill(category);
    return `<g${fill ? ` fill="${fill}"` : ""}>${layerMarkup(category, config.parts[category], idPrefix)}</g>`;
  }).join("");

  const inset = (VIEWBOX / 2) * (1 - FRAME_SCALE);

  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 ${VIEWBOX} ${VIEWBOX}" fill="none">`,
    disc,
    haloFilter(idPrefix),
    `<g filter="url(#${idPrefix}halo)" transform="translate(${inset} ${inset + FRAME_OFFSET_Y}) scale(${FRAME_SCALE})">${layers}</g>`,
    `</svg>`,
  ].join("");
}
