import { cn } from "@/lib/utils";
import type { CSSProperties } from "react";

const rectangleSVG = `<svg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'><rect width='40' height='40' x='0' y='0' stroke='rgba(0,0,0,0.08)' fill='none' /></svg>`;
const encodedRectangleSVG = encodeURIComponent(rectangleSVG);

const gridStyle = {
  "--tour-grid": `url("data:image/svg+xml,${encodedRectangleSVG}")`,
} as CSSProperties;

interface RectanglesProps {
  className?: string;
}

function Rectangles({ className }: RectanglesProps) {
  return (
    <div
      className={cn(
        "pointer-events-none absolute inset-0 z-0 h-full w-full overflow-hidden",
        className,
      )}
    >
      <div
        // oxlint-disable-next-line design-system/no-arbitrary-values -- decorative hero grid image and its 40px tile, no token
        className="h-full w-full bg-[image:var(--tour-grid)] bg-[length:40px_40px] bg-center bg-repeat"
        style={gridStyle}
      />
    </div>
  );
}

export function TourBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 z-0 h-full w-full overflow-hidden opacity-40 [perspective:1000px] [transform-style:preserve-3d]">
      <Rectangles className="[transform:rotateX(45deg)] [mask-image:linear-gradient(to_top,white,transparent)]" />
      <Rectangles className="[transform:rotateX(-45deg)] [mask-image:linear-gradient(to_bottom,white,transparent)]" />
    </div>
  );
}
