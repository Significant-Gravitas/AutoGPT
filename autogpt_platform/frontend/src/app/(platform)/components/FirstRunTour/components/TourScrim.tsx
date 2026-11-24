"use client";

import { cn } from "@/lib/utils";
import { useId, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import type { Box } from "../helpers";

interface Props {
  boxes: Box[];
  /** Dims the page around the boxes; without it only the rings show. */
  dim?: boolean;
}

const HOLE_RADIUS = 12;

// Takes the clicks so a stray one cannot navigate away mid-tour. Portalled so
// no stacking context in the app shell can lift a popup above it.
export function TourScrim({ boxes, dim = true }: Props) {
  const maskID = `tour-mask-${useId().replace(/[^a-zA-Z0-9_-]/g, "")}`;

  return createPortal(
    <div
      aria-hidden
      data-testid="first-run-tour-scrim"
      className={cn(
        "fixed inset-0 z-tour-scrim",
        dim ? "pointer-events-auto" : "pointer-events-none",
      )}
    >
      {dim && (
        <svg className="absolute inset-0 h-full w-full">
          <defs>
            <mask id={maskID}>
              {/* oxlint-disable-next-line design-system/no-raw-colors -- mask luminance, not a colour: white keeps the dim */}
              <rect width="100%" height="100%" fill="white" />
              {boxes.map((box, index) => (
                <rect
                  key={index}
                  x={box.left}
                  y={box.top}
                  width={box.width}
                  height={box.height}
                  rx={HOLE_RADIUS}
                  // oxlint-disable-next-line design-system/no-raw-colors -- mask luminance, not a colour: black cuts the hole
                  fill="black"
                />
              ))}
            </mask>
          </defs>
          <rect
            width="100%"
            height="100%"
            className="fill-scrim/55"
            mask={`url(#${maskID})`}
          />
        </svg>
      )}
      {boxes.map((box, index) => (
        <div
          key={index}
          className={cn(
            "absolute top-[var(--box-top)] left-[var(--box-left)] h-[var(--box-height)] w-[var(--box-width)] rounded-xl transition-[top,left,width,height] duration-300 ease-out motion-reduce:transition-none",
            // Over the dim the lit hole is the highlight; the edge only
            // crisps it. On an undimmed page a dark ring has to carry it.
            dim
              ? "shadow-spotlight"
              : "shadow-drop ring-2 ring-foreground-strong ring-offset-2 ring-offset-surface",
          )}
          style={
            {
              "--box-top": `${box.top}px`,
              "--box-left": `${box.left}px`,
              "--box-width": `${box.width}px`,
              "--box-height": `${box.height}px`,
            } as CSSProperties
          }
        />
      ))}
    </div>,
    document.body,
  );
}
