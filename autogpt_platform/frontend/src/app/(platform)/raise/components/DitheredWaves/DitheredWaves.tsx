"use client";

import { cn } from "@/lib/utils";
import { DITHER_COLORS } from "./helpers";
import { useDitheredWaves } from "./useDitheredWaves";

type Props = {
  colors?: readonly string[];
  className?: string;
};

export function DitheredWaves({ colors = DITHER_COLORS, className }: Props) {
  const canvasRef = useDitheredWaves(colors);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className={cn("block h-full w-full bg-muted", className)}
    />
  );
}
