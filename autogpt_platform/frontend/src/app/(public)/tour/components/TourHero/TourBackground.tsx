import { cn } from "@/lib/utils";
import type { CSSProperties } from "react";

const rectangleSVG = `<svg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'><rect width='40' height='40' x='0' y='0' stroke='rgba(0,0,0,0.08)' fill='none' /></svg>`;
const encodedRectangleSVG = encodeURIComponent(rectangleSVG);

interface RectanglesProps {
  className?: string;
  style?: CSSProperties;
}

function Rectangles({ className, style }: RectanglesProps) {
  return (
    <div
      className={cn(
        "pointer-events-none absolute inset-0 z-0 h-full w-full overflow-hidden",
        className,
      )}
      style={style}
    >
      <div
        className="h-full w-full"
        style={{
          backgroundImage: `url("data:image/svg+xml,${encodedRectangleSVG}")`,
          backgroundSize: "40px 40px",
          backgroundPosition: "center",
          backgroundRepeat: "repeat",
        }}
      />
    </div>
  );
}

export function TourBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 z-0 h-full w-full overflow-hidden [perspective:1000px] [transform-style:preserve-3d]">
      <Rectangles
        style={{ transform: "rotateX(45deg)" }}
        className="[mask-image:linear-gradient(to_top,white,transparent)]"
      />
      <Rectangles
        style={{ transform: "rotateX(-45deg)" }}
        className="[mask-image:linear-gradient(to_bottom,white,transparent)]"
      />
    </div>
  );
}
