// The dial circle hangs above the avatar: its lowest point coincides with the
// avatar, so whichever persona rotates down to the bottom is the selected one.
// Wide enough that full-size (128px) avatars fit the rim with breathing room;
// the top of the ring may run off screen.
export const DIAL_RADIUS = 380;

/** Half of a rim avatar (size-32) — the drag surface extends this far past the rim. */
export const DIAL_ITEM_RADIUS = 64;

// Fixed angular gap between neighbours, independent of how many personas there
// are. The wheel scrolls infinitely: slots are "virtual" indices that extend
// forever in both directions and map onto the roster modulo its length, so
// only the handful of slots near the bottom ever exist in the DOM.
export const DIAL_STEP = 32;

/** Virtual slots rendered either side of the bottom point. */
export const DIAL_WINDOW = 5;

/**
 * Rosters bigger than the render window wrap around forever; smaller ones
 * (e.g. filtered search results) are shown once each on a bounded arc, so the
 * same persona never appears twice.
 */
export function shouldWrap(count: number) {
  return count > DIAL_WINDOW * 2;
}

/** Bounded-arc rotation limits: first persona at one end, last at the other. */
export function clampRotation(rotation: number, count: number) {
  return Math.min(0, Math.max(-(count - 1) * DIAL_STEP, rotation));
}

/** Angle of a pointer around the dial centre, in degrees. */
export function angleFromCentre(
  centre: { x: number; y: number },
  point: { x: number; y: number },
) {
  return (Math.atan2(point.y - centre.y, point.x - centre.x) * 180) / Math.PI;
}

/** Maps a virtual slot onto a roster index. */
export function wrapIndex(virtual: number, count: number) {
  return ((virtual % count) + count) % count;
}

/** The virtual slot closest to the bottom of the dial. */
export function virtualFromRotation(rotation: number) {
  return Math.round(-rotation / DIAL_STEP);
}

/** Which persona currently sits at the bottom of the dial. */
export function indexFromRotation(rotation: number, count: number) {
  return wrapIndex(virtualFromRotation(rotation), count);
}

/** Rotation that parks the given virtual slot at the bottom. */
export function rotationForVirtual(virtual: number) {
  return -virtual * DIAL_STEP;
}

/** The virtual slot for a roster index that is nearest the current rotation. */
export function nearestVirtual(index: number, count: number, rotation: number) {
  const current = -rotation / DIAL_STEP;
  const turns = Math.round((current - index) / count);
  return index + turns * count;
}

export function snapRotation(rotation: number) {
  return Math.round(rotation / DIAL_STEP) * DIAL_STEP;
}

/** Distance in steps from the bottom selection point. */
export function stepsFromBottom(virtual: number, rotation: number) {
  return Math.abs(virtual + rotation / DIAL_STEP);
}
