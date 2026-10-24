"use client";

import { cn } from "@/lib/utils";
import { useId } from "react";
import { createPortal } from "react-dom";
import type { Box } from "../helpers";

interface Props {
  boxes: Box[];
  /** Dims the page around the boxes; without it only the rings show. */
  dim?: boolean;
}

const HOLE_RADIUS = 12;

// Dims the app and cuts a window around the real UI being pointed at. The
// page stays visible on purpose: the whole point is to see where things are.
// It also takes the clicks, so a stray one cannot navigate away mid-tour.
// Portalled to the body so no stacking context in the app shell can lift a
// popup (the changelog, a banner) above it.
export function TourScrim({ boxes, dim = true }: Props) {
  const maskID = `tour-mask-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;

  return createPortal(
    <div
      aria-hidden
      data-testid="first-run-tour-scrim"
      className={cn(
        "fixed inset-0 z-[110]",
        dim ? "pointer-events-auto" : "pointer-events-none",
      )}
    >
      {dim && (
        <svg className="absolute inset-0 h-full w-full">
          <defs>
            <mask id={maskID}>
              <rect width="100%" height="100%" fill="white" />
              {boxes.map((box, index) => (
                <rect
                  key={index}
                  x={box.left}
                  y={box.top}
                  width={box.width}
                  height={box.height}
                  rx={HOLE_RADIUS}
                  fill="black"
                />
              ))}
            </mask>
          </defs>
          <rect
            width="100%"
            height="100%"
            className="fill-zinc-950/50"
            mask={`url(#${maskID})`}
          />
        </svg>
      )}
      {boxes.map((box, index) => (
        <div
          key={index}
          className="absolute rounded-xl shadow-[0_0_0_2px_theme(colors.violet.500),0_0_0_6px_theme(colors.violet.500/0.25)] transition-[top,left,width,height] duration-300 ease-out motion-reduce:transition-none"
          style={{
            top: box.top,
            left: box.left,
            width: box.width,
            height: box.height,
          }}
        />
      ))}
    </div>,
    document.body,
  );
}
