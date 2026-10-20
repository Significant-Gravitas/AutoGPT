import { cn } from "@/lib/utils";
import styles from "./PixelGridLoader.module.css";

// Chevron wavefront: cells light up in diagonal bands driving left to right.
// The cycle is shorter than the sweep, so two fronts are always in flight.
const CHEVRON_DELAYS = Array.from({ length: 9 }, (_, i) => {
  const row = Math.floor(i / 3);
  const column = i % 3;
  return (column + Math.abs(row - 1)) * 90;
});

// Comet lapping the perimeter; the centre cell never lights up.
const ORBIT_ORDER = [0, 1, 2, 5, 8, 7, 6, 3];
const ORBIT_DELAYS = Array.from({ length: 9 }, (_, i) => {
  const step = ORBIT_ORDER.indexOf(i);
  return step === -1 ? null : step * 110;
});

const VARIANTS = {
  drive: { delays: CHEVRON_DELAYS, durationMs: 650, round: false },
  dots: { delays: CHEVRON_DELAYS, durationMs: 650, round: true },
  orbit: { delays: ORBIT_DELAYS, durationMs: 950, round: false },
};

interface Props {
  variant?: keyof typeof VARIANTS;
  /** Edge length of a single cell in pixels. */
  cellSize?: number;
  className?: string;
}

export function PixelGridLoader({
  variant = "drive",
  cellSize = 4,
  className,
}: Props) {
  const { delays, durationMs, round } = VARIANTS[variant];

  return (
    <span
      aria-hidden
      className={cn("grid shrink-0", className)}
      style={{
        gridTemplateColumns: `repeat(3, ${cellSize}px)`,
        gap: cellSize * 0.375,
      }}
    >
      {delays.map((delay, index) => (
        <span
          key={index}
          className={cn(
            "bg-current",
            round ? "rounded-full" : "rounded-[1px]",
            delay !== null && styles.cell,
          )}
          style={{
            width: cellSize,
            height: cellSize,
            opacity: delay === null ? 0.07 : 0.15,
            animationDuration: `${durationMs}ms`,
            animationDelay: `${delay ?? 0}ms`,
          }}
        />
      ))}
    </span>
  );
}
