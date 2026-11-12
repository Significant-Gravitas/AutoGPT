import { HEAD_STONES } from "./stoneStackShapes";

/**
 * Timing for StoneStackLoader, as the `values` / `keyTimes` / `keySplines`
 * strings SVG animation elements take. The loader uses those elements rather
 * than CSS keyframes because Safari does not animate the CSS `d` property,
 * so a CSS morph would leave the head frozen there.
 */
export interface Track {
  values: string;
  keyTimes: string;
  keySplines: string;
}

/** One beat: the head hops, changes volume in the air, lands and rests. */
export const BEAT_SECONDS = 1.9;
export const CYCLE_SECONDS = BEAT_SECONDS * HEAD_STONES.length;

/** Moments within a beat, as fractions of it. */
const CHANGE_FROM = 0.04;
const CHANGE_TO = 0.34;
const HOP_PEAK = 0.18;
const LANDING = 0.36;
const SQUASH_PEAK = 0.47;
const SETTLED = 0.58;

const HOLD = "0 0 1 1";
const EASE_OUT = "0.33 1 0.68 1";
const EASE_IN = "0.32 0 0.67 0";
const EASE_IN_OUT = "0.65 0 0.35 1";

/** A track over the whole cycle: each beat holds one head's value, then
 *  eases to the next head's while the head is in the air. */
function acrossHeads(valueOf: (index: number) => string): Track {
  const count = HEAD_STONES.length;
  const stops = [
    { at: 0, value: valueOf(0) },
    ...HEAD_STONES.flatMap((_, index) => [
      { at: (index + CHANGE_FROM) / count, value: valueOf(index) },
      { at: (index + CHANGE_TO) / count, value: valueOf((index + 1) % count) },
    ]),
    { at: 1, value: valueOf(0) },
  ];
  return {
    values: stops.map((stop) => stop.value).join(";"),
    keyTimes: stops.map((stop) => Number(stop.at.toFixed(4))).join(";"),
    keySplines: stops
      .slice(1)
      .map((_, segment) => (segment % 2 === 1 ? EASE_IN_OUT : HOLD))
      .join(";"),
  };
}

export const HEAD_MORPH = acrossHeads((index) => HEAD_STONES[index].d);
export const HEAD_TILT = acrossHeads((index) => `${HEAD_STONES[index].tilt}`);

export const HEAD_HOP: Track = {
  values: "0 0;0 -2.6;0 0;0 0",
  keyTimes: `0;${HOP_PEAK};${LANDING};1`,
  keySplines: [EASE_OUT, EASE_IN, HOLD].join(";"),
};

const LANDING_KEY_TIMES = `0;${LANDING};${SQUASH_PEAK};${SETTLED};1`;
const LANDING_KEY_SPLINES = [HOLD, EASE_OUT, EASE_IN_OUT, HOLD].join(";");

export const HEAD_SQUASH: Track = {
  values: "1 1;1 1;1.05 0.91;1 1;1 1",
  keyTimes: LANDING_KEY_TIMES,
  keySplines: LANDING_KEY_SPLINES,
};

export const BASE_PRESS: Track = {
  values: "1 1;1 1;1.03 0.95;1 1;1 1",
  keyTimes: LANDING_KEY_TIMES,
  keySplines: LANDING_KEY_SPLINES,
};
