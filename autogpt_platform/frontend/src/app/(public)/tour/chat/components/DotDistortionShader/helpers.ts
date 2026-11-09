/**
 * The colour a canvas can paint for `color`. A `var(--name)` is read from
 * `element`'s computed style; a colour role is an RGB triplet
 * (src/app/tokens/theme.css), which canvas only takes inside `rgb()`.
 */
export function resolveCanvasColour(color: string, element: Element) {
  const name = /^var\(([^)]+)\)$/.exec(color)?.[1];
  if (!name) return color;
  const value = getComputedStyle(element).getPropertyValue(name).trim();
  return /^\d+ \d+ \d+$/.test(value) ? `rgb(${value})` : value;
}
