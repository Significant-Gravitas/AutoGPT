"use client";

import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

import { takeMicLevel } from "../micLevel";

/** One column per tick; the strip scrolls left as they arrive. */
const COLUMNS = 48;
export const TICK_MS = 100;
const MIN_SCALE = 0.06;
/** Speech sits near 0.05–0.2 RMS, so the strip needs the gain to fill. */
const MIC_GAIN = 4;

interface Props {
  source: "mic" | "pulse";
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
  const [columns, setColumns] = useState<number[]>(() =>
    new Array(COLUMNS).fill(0),
  );

  useEffect(() => {
    let tick = 0;
    const timer = setInterval(() => {
      const next = source === "mic" ? micColumn() : pulseColumn(tick++);
      setColumns((previous) => [...previous.slice(1), next]);
    }, TICK_MS);
    return () => clearInterval(timer);
  }, [source]);

  return (
    <div
      className={cn("flex h-6 items-center gap-[2px]", className)}
      aria-hidden="true"
    >
      {columns.map((level, index) => (
        <span
          key={index}
          className={cn("w-[2px] rounded-full transition-none", color)}
          style={{ height: `${Math.max(MIN_SCALE, level) * 100}%` }}
        />
      ))}
    </div>
  );
}

function micColumn(): number {
  return Math.min(1, takeMicLevel() * MIC_GAIN);
}

/**
 * Nothing is coming in — the model is thinking, or talking. A travelling
 * bump reads as activity without pretending to be a measurement.
 */
function pulseColumn(tick: number): number {
  return 0.25 + 0.45 * (0.5 + 0.5 * Math.sin(tick / 2.2));
}
