// Relative imports: jest has no `@/` alias.
import { mixHex, palette } from "../../lib/expertPalette";
import { palettes, type Scheme } from "../../lib/theme";
import type { Area } from "./helpers";

// How much of the area's colour goes into each surface, per scheme.
const CARD_TINT: Record<Scheme, number> = { light: 7, dark: 12 };
const SEAT_TINT = 8;
const WASH_TINT = 12;
// The profile's number pills and skill tiles: a fill of the colour, and the
// colour taken toward the text for what sits on it.
const ACCENT_FILL: Record<Scheme, number> = { light: 16, dark: 24 };
const ACCENT_INK = 70;

export function areaColor(area: Area) {
  return palette[area].hex;
}

/** An area's card: the card colour with a little of the area's in it. */
export function areaTint(area: Area, scheme: Scheme) {
  return mixHex(areaColor(area), CARD_TINT[scheme], palettes[scheme].panelCard);
}

/** The fill of an open seat for the area, on the team card. */
export function seatTint(area: Area, scheme: Scheme) {
  return mixHex(areaColor(area), SEAT_TINT, palettes[scheme].panelCard);
}

/** The top of the profile's wash, which fades down into the page grey. */
export function washTint(area: Area, scheme: Scheme) {
  return mixHex(areaColor(area), WASH_TINT, palettes[scheme].panelBackground);
}

/** The fill behind an accent mark on the profile, from any palette hex. */
export function accentFill(hex: string, scheme: Scheme) {
  return mixHex(hex, ACCENT_FILL[scheme], palettes[scheme].panelCard);
}

/** The accent's colour for a number or icon on `accentFill`. */
export function accentInk(hex: string, scheme: Scheme) {
  return mixHex(hex, ACCENT_INK, palettes[scheme].text);
}
