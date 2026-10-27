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

// Takes the clicks so a stray one cannot navigate away mid-tour. Portalled so
// no stacking context in the app shell can lift a popup above it.
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
            className="fill-zinc-950/55"
            mask={`url(#${maskID})`}
          />
        </svg>
      )}
      {boxes.map((box, index) => (
        <div
          key={index}
          className={cn(
            "absolute rounded-xl transition-[top,left,width,height] duration-300 ease-out motion-reduce:transition-none",
            // Over the dim the lit hole is the highlight; the edge only
            // crisps it. On an undimmed page a dark ring has to carry it.
            dim
              ? "shadow-[0_0_0_1.5px_rgb(255_255_255/0.9),0_0_0_6px_rgb(255_255_255/0.14)]"
              : "shadow-[0_0_0_2px_rgb(var(--color-surface)),0_0_0_4px_rgb(var(--color-foreground-strong)),0_8px_24px_-8px_rgb(0_0_0/0.25)]",
          )}
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
