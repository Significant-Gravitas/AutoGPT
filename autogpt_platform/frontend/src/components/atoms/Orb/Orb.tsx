import { cn } from "@/lib/utils";
import type { CSSProperties } from "react";
import { ORB_STAGE, type OrbVariant, getOrbCells } from "./helpers";

interface Props {
  /** Which sweep the lattice runs: radial, diagonal, ring, column, scattered. */
  variant?: OrbVariant;
  /** Rendered edge length in px; the 28px geometry scales to fit. */
  size?: number;
  label?: string;
  className?: string;
}

/**
 * A 3×3 lattice of dots pulsing in sequence — the "still working" indicator.
 * The dots are drawn in `currentColor`, so callers tint it by setting a text
 * color (an expert's accent, or neutral zinc by default).
 */
export function Orb({
  variant = "S1",
  size = 16,
  label = "Working",
  className,
}: Props) {
  return (
    <span
      role="img"
      aria-label={label}
      className={cn("relative block shrink-0 overflow-hidden", className)}
      style={{ width: size, height: size }}
    >
      <span
        className="absolute left-0 top-0 origin-top-left"
        style={
          {
            width: ORB_STAGE,
            height: ORB_STAGE,
            // The 3 dots on a 6px pitch measure 15px, so the grid is nudged
            // to sit centred on the 28px stage rather than filling it.
            transform: `scale(${size / ORB_STAGE}) translate(6.5px, 6.5px)`,
          } as CSSProperties
        }
      >
        {getOrbCells(variant).map((cell) => (
          <span
            key={cell.key}
            className={cn(
              "absolute size-[3px] rounded-full bg-current opacity-[0.3] motion-reduce:animate-none",
              cell.still ? "opacity-[0.16]" : "animate-orb-wave",
            )}
            style={{
              left: cell.left,
              top: cell.top,
              animationDelay: `${cell.delay}ms`,
            }}
          />
        ))}
      </span>
    </span>
  );
}
