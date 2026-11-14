"use client";

import { cn } from "@/lib/utils";
import { type CSSProperties, useState } from "react";
import type { TrailState } from "./trailSteps";

/* The trail is drawn from counts alone, so every row has a fixed height.
 * Change these only together with the classes that use them. */
export const ROW = 32;
export const SUB = 24;
export const PAD = 2;
/** The marker's centre and the sub-steps' line, from the row's padding. */
const X = 8;
const XS = 20;
const BEND = 14;
const BEND_LENGTH = 20;
/** The light: thin slices, faint at the back and full at the front. */
const SLICES = 10;
const SLICE = 2;

export const BEND_OUT = `M${X} 0 C${X} 8 ${XS} 6 ${XS} ${BEND}`;
export const BEND_BACK = `M${XS} 0 C${XS} 8 ${X} 6 ${X} ${BEND}`;

const MARK = "relative mr-3.5 grid size-4 shrink-0 place-items-center";

interface MarkerProps {
  state: TrailState;
}

/** A step's mark on the trail. A step that changes while you watch plays
 *  its mark in; one that was already so when the task opened just shows. */
export function Marker({ state }: MarkerProps) {
  const [first] = useState(state);
  const isFresh = state !== first;

  if (state === "done") {
    return (
      <span aria-hidden className={MARK}>
        <span
          className={cn(
            "absolute inset-0 rounded-full bg-info/12",
            isFresh && "animate-trail-pop",
          )}
        />
        <svg
          viewBox="0 0 16 16"
          className="relative size-4 fill-none stroke-info"
          strokeWidth={1.7}
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path
            d="M4.9 8.2 7.1 10.3 11.1 5.9"
            pathLength={1}
            strokeDasharray={1}
            className={isFresh ? "animate-trail-draw" : undefined}
          />
        </svg>
      </span>
    );
  }
  if (state === "running") {
    return (
      <span aria-hidden className={MARK}>
        <svg
          viewBox="0 0 16 16"
          className="size-4 animate-trail-arc fill-none"
          strokeWidth={1.5}
          strokeLinecap="round"
        >
          <circle cx="8" cy="8" r="6.25" className="stroke-info/15" />
          <circle
            cx="8"
            cy="8"
            r="6.25"
            pathLength={40}
            strokeDasharray="12 28"
            className="stroke-info"
          />
        </svg>
      </span>
    );
  }
  if (state === "failed") {
    return (
      <span aria-hidden className={MARK}>
        <span
          className={cn(
            "absolute inset-0 rounded-full bg-error/12",
            isFresh && "animate-trail-pop",
          )}
        />
        <svg
          viewBox="0 0 16 16"
          className="relative size-4 fill-none stroke-error"
          strokeWidth={1.7}
          strokeLinecap="round"
        >
          <path d="M6 6l4 4M10 6l-4 4" />
        </svg>
      </span>
    );
  }
  return (
    <span aria-hidden className={MARK}>
      <svg viewBox="0 0 16 16" className="size-4" strokeWidth={1.5}>
        <circle
          cx="8"
          cy="8"
          r="6.25"
          className={
            state === "waiting"
              ? "fill-status-waiting/15 stroke-status-waiting"
              : "fill-none stroke-foreground-ink/10"
          }
        />
      </svg>
    </span>
  );
}

interface DotProps {
  kind: "done" | "current" | "fold";
}

/** A sub-step's dot. The one being worked on rings as the light lands. */
export function Dot({ kind }: DotProps) {
  if (kind === "current") {
    return (
      <span
        aria-hidden
        className="relative size-[5px] shrink-0 rounded-full bg-info"
      >
        <span className="absolute -inset-[3px] animate-trail-ring rounded-full opacity-0 ring-2 ring-info" />
      </span>
    );
  }
  return (
    <span
      aria-hidden
      className={cn(
        "size-[5px] shrink-0 rounded-full",
        kind === "fold"
          ? "border border-foreground-faint"
          : "bg-foreground-faint",
      )}
    />
  );
}

interface LightProps {
  /** Which sub-step row the light runs into, counted from the top. */
  index: number;
}

/** The light that runs from a working step's mark down its branch into the
 *  sub-step it is on: slices of one path that move together, each fainter
 *  than the one ahead. Past the path's end a slice is not drawn, so the
 *  light drains into the dot. */
export function Light({ index }: LightProps) {
  const end = ROW / 2 - 8 + PAD + SUB / 2 + SUB * index;
  const path = `${BEND_OUT} L${XS} ${end}`;
  const travel = BEND_LENGTH + end - BEND + SLICES * SLICE;

  return (
    <svg
      aria-hidden
      viewBox={`0 0 28 ${end + 2}`}
      width={28}
      height={end + 2}
      className="pointer-events-none absolute top-6 left-2 overflow-visible fill-none stroke-info motion-reduce:hidden"
      strokeWidth={1.5}
      style={{ "--trail-len": travel } as CSSProperties}
    >
      {Array.from({ length: SLICES }, (_, slice) => (
        <path
          key={slice}
          d={path}
          strokeDasharray={`${SLICE} 9999`}
          strokeOpacity={(slice + 1) / SLICES}
          className="animate-trail-light"
          style={{ "--trail-k": (SLICES - slice) * SLICE } as CSSProperties}
        />
      ))}
    </svg>
  );
}
