"use client";

import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

import { createLevelScale } from "../levelScale";
import { takeMicLevel } from "../micLevel";
import { readSpeechLevel } from "../speechLevel";

/** One column per tick; the strip scrolls left as they arrive. */
const COLUMNS = 60;
export const TICK_MS = 67;
const MIN_SCALE = 0.06;
const PULSE_PEAK = 0.7;
/** In ticks per radian — about one breath every 1.4 s. */
const PULSE_PERIOD = 3.3;

export type TraceSource = "mic" | "speech" | "pulse";

interface Column {
  level: number;
  /** The colour of the stage this column was recorded in. */
  color: string;
}

interface Props {
  source: TraceSource;
  className?: string;
  /** Bar colour, as a Tailwind background class. */
  color: string;
}

/**
 * A rolling level meter: a new column every {@link TICK_MS}, older ones
 * shifting left. It replaces the status word — the shape says whether the
 * mic is open, whether it is hearing anything, and how loud.
 */
export function VoiceTrace({ source, className, color }: Props) {
  // Each column keeps the colour of the stage it was recorded in, so the
  // handover shows as black scrolling off while green fills in — not the
  // whole history repainting the instant the mic reopens.
  const [columns, setColumns] = useState<Column[]>(() =>
    new Array(COLUMNS).fill({ level: 0, color }),
  );

  useEffect(() => {
    let tick = 0;
    const scale = createLevelScale();
    const timer = setInterval(() => {
      // Read OUTSIDE the updater. Taking a level consumes it, and React
      // re-invokes updaters (StrictMode does it every time in dev), so the
      // second call saw an already-emptied peak — a flat line with the odd
      // spike when a frame landed between the two.
      const next = { level: column(source, tick++, scale), color };
      setColumns((previous) => [...previous.slice(1), next]);
    }, TICK_MS);
    return () => clearInterval(timer);
  }, [source, color]);

  return (
    <div
      className={cn(
        "flex h-6 items-center justify-center gap-[3px] overflow-hidden",
        className,
      )}
      aria-hidden="true"
    >
      {columns.map(({ level, color: recorded }, index) => (
        <span
          key={index}
          className={cn("w-[3px] shrink-0 rounded-full", recorded)}
          style={{ height: `${Math.max(MIN_SCALE, level) * 100}%` }}
        />
      ))}
    </div>
  );
}

function column(
  source: TraceSource,
  tick: number,
  scale: (level: number) => number,
): number {
  if (source === "mic") return scale(takeMicLevel());
  if (source === "speech") {
    const level = readSpeechLevel();
    // No analyser — Web Audio unavailable, or the element was already routed
    // elsewhere. Falling through to the pulse beats a dead flat line.
    if (level !== null) return scale(level);
  }
  return pulse(tick);
}

/**
 * Nothing to measure — the model is thinking. A travelling bump reads as
 * activity without pretending to be a measurement. It starts from zero so
 * it grows out of the flat line the mic left behind rather than jumping.
 */
function pulse(tick: number): number {
  return PULSE_PEAK * (0.5 - 0.5 * Math.cos(tick / PULSE_PERIOD));
}
