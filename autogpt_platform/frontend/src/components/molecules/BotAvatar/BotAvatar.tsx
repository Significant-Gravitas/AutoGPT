"use client";

import { cn } from "@/lib/utils";
import { useId, useRef } from "react";
import { Accessory } from "./components/Accessory";
import { Face } from "./components/Face";
import { StatusBadge } from "./components/StatusBadge";
import {
  encodeConfig,
  findColor,
  findShape,
  INK,
  mixHex,
  VIEWBOX,
  type AvatarConfig,
  type AvatarStatus,
} from "./helpers";
import {
  ellipsoidFor,
  foreshortenTransform,
  isVisible,
  project,
  surfacePointAt,
  type Pose,
} from "./projection";
import { usePose } from "./usePose";

interface Props {
  config: AvatarConfig;
  status?: AvatarStatus;
  size?: number;
  animated?: boolean;
  trackPointer?: boolean;
  poseOffset?: Partial<Pose>;
  outline?: boolean;
  showBadge?: boolean;
  title?: string;
  className?: string;
}

const SPOTS = [
  { dx: -0.28, dy: 14, r: 2.4 },
  { dx: -0.2, dy: 9, r: 1.5 },
  { dx: -0.33, dy: 22, r: 1.3 },
];

export function BotAvatar({
  config,
  status = "idle",
  size = 96,
  animated = true,
  trackPointer = false,
  poseOffset,
  outline = false,
  showBadge = true,
  title,
  className,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const clipId = useId();
  const shape = findShape(config.shape);
  const color = findColor(config.color);
  const { isLive, isBlinking, pose } = usePose({
    status,
    animated,
    trackPointer,
    poseOffset,
    svgRef,
  });
  const { cx, top, bottom, width } = shape.anchors;
  const body = ellipsoidFor(shape.anchors);
  const rollDeg = (pose.roll * 180) / Math.PI;
  const bodyFill = outline ? color.body : mixHex(color.body, color.mid, 0.35);
  const shadeFill = outline ? color.mid : mixHex(color.mid, color.deep, 0.18);

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${VIEWBOX} ${VIEWBOX}`}
      width={size}
      height={size}
      role="img"
      aria-label={
        title ?? `${color.label} ${shape.label.toLowerCase()} avatar, ${status}`
      }
      data-testid="bot-avatar"
      data-avatar={encodeConfig(config)}
      data-status={status}
      data-outline={outline}
      className={cn("shrink-0 overflow-visible", className)}
    >
      <defs>
        <clipPath id={clipId}>
          <path d={shape.path} />
        </clipPath>
      </defs>
      <g
        transform={`translate(0 ${-pose.bob}) rotate(${rollDeg} ${cx} ${bottom})`}
      >
        <path
          d={shape.path}
          fill={bodyFill}
          stroke={outline ? INK : "none"}
          strokeWidth={3}
          strokeLinejoin="round"
        />
        <g clipPath={`url(#${clipId})`}>
          <ellipse
            cx={cx}
            cy={bottom + 8 + pose.pitch * 6}
            rx={width * 0.6}
            ry={22}
            fill={shadeFill}
            opacity={0.75}
          />
          {SPOTS.map((spot, index) => {
            const point = project(
              surfacePointAt(cx + width * spot.dx, top + spot.dy, body),
              pose,
              body,
            );
            return isVisible(point) ? (
              <circle
                key={index}
                transform={foreshortenTransform(point)}
                r={spot.r}
                fill={shadeFill}
              />
            ) : null;
          })}
        </g>
        <Face
          anchors={shape.anchors}
          pose={pose}
          status={status}
          blush={shadeFill}
          isLive={isLive}
          isBlinking={isBlinking}
        />
        <Accessory
          accessory={config.accessory}
          anchors={shape.anchors}
          pose={pose}
          deep={color.deep}
          outline={outline}
        />
      </g>
      {showBadge ? (
        <StatusBadge status={status} anchors={shape.anchors} isLive={isLive} />
      ) : null}
    </svg>
  );
}
