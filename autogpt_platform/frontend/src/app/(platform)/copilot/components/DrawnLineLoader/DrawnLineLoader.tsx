import { cn } from "@/lib/utils";
import styles from "./DrawnLineLoader.module.css";
import { DRAWN_LINE_SHAPES, DRAWN_LINE_VIEWBOX } from "./drawnLineShapes";

/** Time each shape owns the stage, including its crossfade. */
const SHAPE_SECONDS = 2.5;

interface Props {
  className?: string;
  /** Edge length of the square loader in pixels. */
  size?: number;
}

/**
 * A single line that draws itself into loose, hand-drawn shapes, one after
 * another. Pure SVG + CSS keyframes (see the module css), no JS animation.
 * Colour comes from `currentColor`. Under reduced motion it shows the first
 * shape, fully drawn and still.
 */
export function DrawnLineLoader({ className, size = 20 }: Props) {
  const cycleSeconds = DRAWN_LINE_SHAPES.length * SHAPE_SECONDS;

  return (
    <svg
      aria-hidden
      focusable="false"
      viewBox={`0 0 ${DRAWN_LINE_VIEWBOX} ${DRAWN_LINE_VIEWBOX}`}
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={cn("block shrink-0", className)}
    >
      {DRAWN_LINE_SHAPES.map((shape, index) => (
        <path
          key={shape.name}
          d={shape.d}
          pathLength={1}
          className={styles.path}
          style={{
            animationDuration: `${cycleSeconds}s`,
            animationDelay: `${index * SHAPE_SECONDS}s`,
          }}
        />
      ))}
    </svg>
  );
}
