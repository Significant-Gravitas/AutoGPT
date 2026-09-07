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
  VIEWBOX,
  type AvatarConfig,
  type AvatarStatus,
} from "./helpers";
import type { Pose } from "./projection";
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
  const ids = useId().replace(/:/g, "");
  const clipId = `${ids}-clip-${config.shape}`;
  const gradientId = `${ids}-body-${config.color}`;
  const shape = findShape(config.shape);
  const color = findColor(config.color);
  const { isLive, isBlinking, pose } = usePose({
    status,
    animated,
    trackPointer,
    poseOffset,
    svgRef,
  });
  const { cx, bottom } = shape.anchors;
  const rollDeg = (pose.roll * 180) / Math.PI;
  const bodyFill = outline ? color.body : `url(#${gradientId})`;

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
        <radialGradient id={gradientId} cx="50%" cy="45%" r="62%">
          <stop offset="0%" stopColor={color.body} />
          <stop offset="100%" stopColor={color.light} />
        </radialGradient>
      </defs>
      <g
        transform={`translate(0 ${-pose.bob}) rotate(${rollDeg} ${cx} ${bottom})`}
      >
        <Accessory
          accessory={config.accessory}
          anchors={shape.anchors}
          pose={pose}
          deep={color.deep}
          outline={outline}
          layer="back"
        />
        <path
          d={shape.path}
          fill={bodyFill}
          stroke={outline ? INK : "none"}
          strokeWidth={3}
          strokeLinejoin="round"
        />
        <Face
          anchors={shape.anchors}
          pose={pose}
          status={status}
          blush={color.mid}
          isLive={isLive}
          isBlinking={isBlinking}
        />
        <Accessory
          accessory={config.accessory}
          anchors={shape.anchors}
          pose={pose}
          deep={color.deep}
          outline={outline}
          layer="front"
        />
      </g>
      {showBadge ? (
        <StatusBadge status={status} anchors={shape.anchors} isLive={isLive} />
      ) : null}
    </svg>
  );
}
