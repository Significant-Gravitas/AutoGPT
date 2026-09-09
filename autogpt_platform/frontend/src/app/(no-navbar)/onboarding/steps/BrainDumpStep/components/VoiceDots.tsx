"use client";

import { motion, type MotionValue, useTransform } from "framer-motion";

export const DOT = 14;
const GAP = 22;
const BAR_HEIGHTS = [40, 64, 64, 40];
// Which voice band each dot listens to, low to high.
const BANDS = [0, 1, 3, 4];

interface Props {
  levels: MotionValue<number>[];
  // 0: the body is whole; 1: it has collapsed into one dot.
  collapse: MotionValue<number>;
  // 0: one dot; 1: four dots spread into a row.
  split: MotionValue<number>;
  centre: { x: number; y: number };
  color: string;
  reduceMotion: boolean;
}

// The row of dots the body turns into while listening: they sit stacked as
// one dot until `split` spreads them, then each stretches with its band of
// the voice, like a voice-memo icon.
export function VoiceDots({
  levels,
  collapse,
  split,
  centre,
  color,
  reduceMotion,
}: Props) {
  return (
    <>
      {BANDS.map((band, index) => (
        <Dot
          key={band}
          level={levels[band]}
          collapse={collapse}
          split={split}
          slot={index - (BANDS.length - 1) / 2}
          barHeight={BAR_HEIGHTS[index]}
          centre={centre}
          color={color}
          reduceMotion={reduceMotion}
        />
      ))}
    </>
  );
}

function Dot({
  level,
  collapse,
  split,
  slot,
  barHeight,
  centre,
  color,
  reduceMotion,
}: {
  level: MotionValue<number>;
  collapse: MotionValue<number>;
  split: MotionValue<number>;
  slot: number;
  barHeight: number;
  centre: { x: number; y: number };
  color: string;
  reduceMotion: boolean;
}) {
  const x = useTransform(split, (spread) => slot * GAP * spread);
  const opacity = useTransform(collapse, [0.45, 1], [0, 1]);
  const scale = useTransform(collapse, [0.45, 1], [0.4, 1]);
  const height = useTransform(
    [level, split],
    ([voice, spread]: number[]) =>
      DOT + (reduceMotion ? 0 : voice * spread * (barHeight - DOT)),
  );

  return (
    <motion.span
      data-testid="voice-dot"
      aria-hidden
      className="absolute rounded-full"
      style={{
        width: DOT,
        height,
        left: centre.x - DOT / 2,
        top: centre.y - DOT / 2,
        x,
        y: useTransform(height, (h) => (DOT - h) / 2),
        scale,
        opacity,
        backgroundColor: color,
        willChange: "transform, height",
      }}
    />
  );
}
