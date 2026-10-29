import type { AvatarStatus } from "./status";

export const BADGE_FILL: Record<Exclude<AvatarStatus, "idle">, string> = {
  thinking: "#7C3AED",
  working: "#3B6FD1",
  waiting: "#E8A317",
  done: "#22A05B",
  failed: "#DC2626",
  sleeping: "#71717A",
};

// Drawn at a radius of 8 and scaled onto the avatar's much larger canvas.
const BADGE_SCALE = 7;
const DIAGONAL = Math.SQRT1_2;

// Glyphs are drawn around the origin at a radius of 8, so the same markup
// serves the in-artwork badge and the overlay one.
export const BADGE_GLYPH: Record<Exclude<AvatarStatus, "idle">, string> = {
  working: '<circle r="3" fill="#fff"/>',
  done: '<path d="M-3.5,0.2 L-1,2.8 L3.8,-2.6" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
  thinking:
    '<g fill="#fff"><circle cx="-3.2" r="1.4"/><circle cx="0" r="1.4"/><circle cx="3.2" r="1.4"/></g>',
  failed:
    '<path d="M-3,-3 L3,3 M3,-3 L-3,3" fill="none" stroke="#fff" stroke-width="2.2" stroke-linecap="round"/>',
  sleeping:
    '<path d="M-3,-3 L3,-3 L-3,3 L3,3" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>',
  waiting:
    '<path d="M0,-3.6 L0,0.8 M0,3.4 L0,3.5" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round"/>',
};

export function badgeMarkup(status: AvatarStatus, viewBox: number): string {
  if (status === "idle") return "";
  const centre = viewBox / 2;
  const offset = centre + centre * DIAGONAL;
  return [
    `<g transform="translate(${offset} ${offset}) scale(${BADGE_SCALE})">`,
    `<circle r="8" fill="${BADGE_FILL[status]}" stroke="#fff" stroke-width="2.5"/>`,
    BADGE_GLYPH[status],
    `</g>`,
  ].join("");
}
