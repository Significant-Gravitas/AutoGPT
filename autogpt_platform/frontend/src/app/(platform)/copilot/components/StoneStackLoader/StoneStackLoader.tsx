"use client";

import { cn } from "@/lib/utils";
import { useReducedMotion } from "framer-motion";
import { useId } from "react";
import {
  AROUND_BASE_BAND,
  BASE_FOOT,
  BASE_STONE,
  HEAD_CONTACT,
  HEAD_STONES,
} from "./stoneStackShapes";
import {
  BASE_PRESS,
  BEAT_SECONDS,
  CYCLE_SECONDS,
  HEAD_HOP,
  HEAD_MORPH,
  HEAD_SQUASH,
  HEAD_TILT,
  type Track,
} from "./timeline";

interface Props {
  className?: string;
  /** Edge length of the square loader in pixels. */
  size?: number;
  /** The material colour of whoever is thinking. Without one the loader
   *  takes the surrounding text colour. */
  color?: string | null;
}

/**
 * An expert without the face, thinking: a head stone on a base stone. Every
 * beat the head hops, changes into the next head volume in the air and lands
 * with a new tilt. Under reduced motion it is the first figure, still.
 */
export function StoneStackLoader({ className, size = 20, color }: Props) {
  const aroundBandID = useId();
  const isStill = useReducedMotion();

  return (
    <svg
      aria-hidden
      focusable="false"
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill={color ?? "currentColor"}
      className={cn("block shrink-0 overflow-visible", className)}
    >
      <clipPath id={aroundBandID}>
        <path d={AROUND_BASE_BAND} clipRule="evenodd" />
      </clipPath>
      <g transform={BASE_FOOT}>
        <g>
          {!isStill && (
            <Move type="scale" track={BASE_PRESS} seconds={BEAT_SECONDS} />
          )}
          <path d={BASE_STONE} fillOpacity={0.5} />
          <path d={BASE_STONE} clipPath={`url(#${aroundBandID})`} />
        </g>
      </g>
      <g transform={HEAD_CONTACT}>
        <g>
          {!isStill && (
            <Move type="translate" track={HEAD_HOP} seconds={BEAT_SECONDS} />
          )}
          <g>
            {!isStill && (
              <Move type="scale" track={HEAD_SQUASH} seconds={BEAT_SECONDS} />
            )}
            <path d={HEAD_STONES[0].d}>
              {!isStill && (
                <>
                  <animate
                    attributeName="d"
                    dur={`${CYCLE_SECONDS}s`}
                    repeatCount="indefinite"
                    calcMode="spline"
                    {...HEAD_MORPH}
                  />
                  <Move
                    type="rotate"
                    track={HEAD_TILT}
                    seconds={CYCLE_SECONDS}
                  />
                </>
              )}
            </path>
          </g>
        </g>
      </g>
    </svg>
  );
}

interface MoveProps {
  type: "translate" | "scale" | "rotate";
  track: Track;
  seconds: number;
}

function Move({ type, track, seconds }: MoveProps) {
  return (
    <animateTransform
      attributeName="transform"
      type={type}
      dur={`${seconds}s`}
      repeatCount="indefinite"
      calcMode="spline"
      {...track}
    />
  );
}
