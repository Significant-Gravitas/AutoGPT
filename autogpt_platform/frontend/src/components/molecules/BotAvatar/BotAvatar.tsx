"use client";

import { cn } from "@/lib/utils";
import {
  motion,
  type TargetAndTransition,
  type Transition,
} from "framer-motion";
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
import { useBotAvatar } from "./useBotAvatar";

interface Props {
  config: AvatarConfig;
  status?: AvatarStatus;
  size?: number;
  animated?: boolean;
  trackPointer?: boolean;
  showBadge?: boolean;
  title?: string;
  className?: string;
}

const BODY_MOTION: Record<
  AvatarStatus,
  { animate: TargetAndTransition; transition: Transition }
> = {
  idle: {
    animate: { y: [0, -1.5, 0], scaleY: [1, 1.02, 1] },
    transition: { duration: 3.4, repeat: Infinity, ease: "easeInOut" },
  },
  working: {
    animate: { y: [0, -3, 0], rotate: [0, -1.5, 0, 1.5, 0] },
    transition: { duration: 0.75, repeat: Infinity, ease: "easeInOut" },
  },
  waiting: {
    animate: { rotate: [0, -5, 0, 5, 0], y: [0, -1, 0, -1, 0] },
    transition: { duration: 2.2, repeat: Infinity, ease: "easeInOut" },
  },
  done: {
    animate: { y: [0, -9, 0, -2, 0], scale: [1, 1.08, 0.97, 1.02, 1] },
    transition: { duration: 0.85, ease: "easeOut" },
  },
};

export function BotAvatar({
  config,
  status = "idle",
  size = 96,
  animated = true,
  trackPointer = false,
  showBadge = true,
  title,
  className,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const clipId = useId();
  const shape = findShape(config.shape);
  const color = findColor(config.color);
  const { isLive, isBlinking, gaze } = useBotAvatar({
    animated,
    trackPointer,
    svgRef,
  });
  const { cx, top, bottom, width } = shape.anchors;
  const body = BODY_MOTION[status];

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
      className={cn("shrink-0 overflow-visible", className)}
    >
      <defs>
        <clipPath id={clipId}>
          <path d={shape.path} />
        </clipPath>
      </defs>
      <motion.g
        key={isLive ? status : "static"}
        style={{ originX: `${cx}px`, originY: `${bottom}px` }}
        animate={isLive ? body.animate : undefined}
        transition={body.transition}
      >
        <path
          d={shape.path}
          fill={color.body}
          stroke={INK}
          strokeWidth={3}
          strokeLinejoin="round"
        />
        <g clipPath={`url(#${clipId})`}>
          <ellipse
            cx={cx}
            cy={bottom + 8}
            rx={width * 0.6}
            ry={22}
            fill={color.mid}
            opacity={0.75}
          />
        </g>
        <g fill={color.mid}>
          <circle cx={cx - width * 0.28} cy={top + 14} r={2.4} />
          <circle cx={cx - width * 0.2} cy={top + 9} r={1.5} />
          <circle cx={cx - width * 0.33} cy={top + 22} r={1.3} />
        </g>
        <Face
          anchors={shape.anchors}
          status={status}
          blush={color.mid}
          isLive={isLive}
          isBlinking={isBlinking}
          gaze={gaze}
        />
        <Accessory
          accessory={config.accessory}
          anchors={shape.anchors}
          deep={color.deep}
        />
      </motion.g>
      {showBadge ? (
        <StatusBadge status={status} anchors={shape.anchors} isLive={isLive} />
      ) : null}
    </svg>
  );
}
