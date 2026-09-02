export type OrbVariant = "S1" | "S2" | "S3" | "S4" | "S5";

/** The stage the geometry is tuned on; the orb scales from here to `size`. */
export const ORB_STAGE = 28;

const N = 3;
/** Centre-to-centre dot spacing, in stage px. */
const PITCH = 6;
const MID = (N - 1) / 2;

/** Clockwise walk of the lattice perimeter — the track the ring sweeps run on. */
const RING: [number, number][] = (() => {
  const ring: [number, number][] = [];
  for (let x = 0; x < N; x++) ring.push([x, 0]);
  for (let y = 1; y < N; y++) ring.push([N - 1, y]);
  for (let x = N - 2; x >= 0; x--) ring.push([x, N - 1]);
  for (let y = N - 2; y >= 1; y--) ring.push([0, y]);
  return ring;
})();

const RING_INDEX = new Map(RING.map(([x, y], i) => [`${x},${y}`, i]));

/**
 * Per-cell animation delay in ms. Every cell runs the same wave; the stagger
 * is what makes nine identical animations read as one moving front. Negative
 * values seed a cell partway into its cycle.
 */
function cellDelay(variant: OrbVariant, x: number, y: number): number {
  const dx = x - MID;
  const dy = y - MID;
  switch (variant) {
    // Radiates from the centre. The centre leads a beat early so the next
    // swell doesn't sit behind the outer fade.
    case "S1":
      return Math.hypot(dx, dy) * 700 - (dx === 0 && dy === 0 ? 180 : 0);
    // A broad band crossing on the diagonal; the spread is close to the wave
    // duration, so the far corner restarts as the near one does.
    case "S2":
      return ((x + y) / (2 * (N - 1))) * 1500;
    // One head with a decaying tail, running the perimeter clockwise.
    case "S3":
      return ringDelay(x, y, (index, length) => (length - index) % length);
    // A soft column travelling left to right.
    case "S4":
      return (x / (N - 1)) * 1100;
    // Like S3, but the pulse jumps pseudo-randomly around the ring.
    case "S5":
      return ringDelay(x, y, (index, length) => (index * 3) % length);
  }
}

function ringDelay(
  x: number,
  y: number,
  order: (index: number, length: number) => number,
): number {
  const index = RING_INDEX.get(`${x},${y}`);
  if (index === undefined) return 0;
  return -(order(index, RING.length) / RING.length) * 1700;
}

export interface OrbCell {
  key: string;
  left: number;
  top: number;
  delay: number;
  /** Sits the choreography out — interior cells during the ring sweeps. */
  still: boolean;
}

/** The nine lattice cells, with their stage position and phase. */
export function getOrbCells(variant: OrbVariant): OrbCell[] {
  const cells: OrbCell[] = [];
  for (let y = 0; y < N; y++) {
    for (let x = 0; x < N; x++) {
      cells.push({
        key: `${x},${y}`,
        left: x * PITCH,
        top: y * PITCH,
        delay: cellDelay(variant, x, y),
        still:
          (variant === "S3" || variant === "S5") &&
          !RING_INDEX.has(`${x},${y}`),
      });
    }
  }
  return cells;
}
