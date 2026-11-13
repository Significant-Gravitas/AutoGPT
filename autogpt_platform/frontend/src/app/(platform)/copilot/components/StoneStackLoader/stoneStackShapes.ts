/**
 * Artwork for StoneStackLoader: an expert without the face. A head stone
 * rests on a base stone, on a 24x24 canvas.
 *
 * The heads are six of the avatar head volumes (see the backend's
 * `avatar_design.py`). Each is drawn around the point where it touches the
 * base, so a tilt or a squash pivots on that point. All six use the same
 * sequence of path commands, which is what lets the browser morph one into
 * the next: keep that sequence when you edit or add a head.
 */
interface HeadStone {
  /** Stable name, used in tests. */
  name: string;
  /** Path data, with the contact point at 0,0 and the stone above it. */
  d: string;
  /** Degrees the head leans while it rests. */
  tilt: number;
}

/** Where the head touches the base, and where the base meets the ground. */
export const HEAD_CONTACT = "translate(12 12.3)";
export const BASE_FOOT = "translate(12 22)";

/** The first head doubles as the still figure shown under reduced motion. */
export const HEAD_STONES: readonly HeadStone[] = [
  {
    name: "pebble",
    tilt: 0,
    d: "M-6.4 -2.1 Q-8 -4.3 -6.72 -6.7 L-6.4 -7.3 Q-4.8 -10.3 -1.4 -10.45 L0.75 -10.55 Q4 -10.7 6.2 -8.3 L6.28 -8.2 Q8.4 -5.9 7 -3.1 L7 -3.1 Q5.6 -0.3 2.47 -0.18 L-2.08 0 Q-4.8 0.1 -6.4 -2.1 Z",
  },
  {
    name: "wedge",
    tilt: -7,
    d: "M-7 -1.53 Q-8.4 -3.13 -6.69 -4.39 L-5.4 -5.33 Q-2.4 -7.53 0.99 -9.07 L4.68 -10.75 Q6.4 -11.53 7.4 -9.93 L7.4 -9.93 Q8.4 -8.33 7.94 -6.5 L7.27 -3.82 Q6.4 -0.33 2.8 -0.21 L-3.48 0 Q-5.6 0.07 -7 -1.53 Z",
  },
  {
    name: "kite",
    tilt: 6,
    d: "M-5.59 -2.77 Q-8.4 -5.02 -5.78 -7.49 L-3.06 -10.05 Q-1.6 -11.42 0.4 -11.22 L0.4 -11.22 Q2.4 -11.02 3.92 -9.71 L5.68 -8.18 Q8.4 -5.82 5.77 -3.36 L3.61 -1.33 Q2 0.18 -0.2 -0.02 L-0.2 -0.02 Q-2.4 -0.22 -4.12 -1.6 Z",
  },
  {
    name: "dome",
    tilt: -4,
    d: "M-7.8 -2.8 Q-8 -5.6 -6.1 -7.66 L-5.6 -8.2 Q-3.2 -10.8 0.33 -10.6 L0.72 -10.58 Q4 -10.4 6 -7.8 L6.41 -7.27 Q8 -5.2 7.8 -2.6 L7.8 -2.6 Q7.6 0 4.99 0 L-4.8 0 Q-7.6 0 -7.8 -2.79 Z",
  },
  {
    name: "saddle",
    tilt: 8,
    d: "M-7.81 -6.48 Q-8.4 -9.2 -5.8 -8.2 L-5.8 -8.2 Q-3.2 -7.2 -0.41 -7.2 L0.34 -7.2 Q3.2 -7.2 5.8 -8.4 L5.8 -8.4 Q8.4 -9.6 7.82 -6.8 L7.22 -3.92 Q6.4 0 2.4 0 L-2.4 0 Q-6.4 0 -7.25 -3.91 Z",
  },
  {
    name: "slab",
    tilt: -6,
    d: "M-8.21 -2.08 Q-8.4 -4 -8.2 -5.92 L-8.2 -5.92 Q-8 -7.84 -6.07 -7.86 L5.99 -7.98 Q8 -8 8.2 -6 L8.2 -6 Q8.4 -4 8.2 -2 L8.2 -2 Q8 0 5.99 0 L-5.99 0 Q-8 0 -8.2 -2 Z",
  },
];

/** Drawn around the middle of its foot. */
export const BASE_STONE =
  "M-3.34 0 Q-5.13 0 -5.41 -1.76 L-5.41 -1.76 Q-5.7 -3.53 -4.88 -5.12 L-4.45 -5.96 Q-3.19 -8.4 -0.45 -8.4 L0.45 -8.4 Q3.19 -8.4 4.45 -5.96 L4.88 -5.12 Q5.7 -3.53 5.41 -1.76 L5.41 -1.76 Q5.13 0 3.34 0 Z";

/**
 * The avatars' cream sweep, which rises from the foot to the right shoulder
 * and runs past the base at both ends. The loader shows it as a lighter band
 * of the same colour, so it needs no colour of its own: the base is drawn
 * once faint and once solid, the solid one clipped to everything around the
 * band. This is that clip, a box around the base with the band cut out
 * (even-odd).
 */
export const AROUND_BASE_BAND =
  "M-7 -10 H7 V2 H-7 Z M-0.41 1.54 Q0.1 -2.86 7.13 -4.85 L6.47 -7.15 Q-2.1 -4.74 -2.79 1.26 Z";
