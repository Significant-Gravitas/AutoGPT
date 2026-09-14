"use client";

import { motion } from "framer-motion";
import dynamic from "next/dynamic";
import { useId, useRef } from "react";
import type { AvatarStatus, Expression } from "./expressions";
import { findNotionColor, type NotionAvatarConfig } from "./helpers";
import type { Pose } from "./pose";
import { useExpression } from "./useExpression";
import { usePose } from "./usePose";

// The artwork is a few hundred kilobytes of paths. Loading it with the route
// would put it on every page that shows an avatar, including the ones that
// only ever draw a resting face — so it arrives as its own chunk when an
// animated avatar actually mounts. Until it lands, the disc stands in.
const NotionAvatarSvg = dynamic(
  () => import("./NotionAvatarSvg").then((module) => module.NotionAvatarSvg),
  {
    ssr: false,
    loading: () => null,
  },
);

interface Props {
  config: NotionAvatarConfig;
  status?: AvatarStatus;
  expression?: Expression;
  size?: number;
  animated?: boolean;
  trackPointer?: boolean;
  poseOffset?: Partial<Pose>;
  showBadge?: boolean;
  transparent?: boolean;
  title?: string;
  className?: string;
}

export function NotionAvatar({
  config,
  status = "idle",
  expression: expressionOverride,
  size = 96,
  animated = true,
  trackPointer = false,
  poseOffset,
  showBadge = true,
  transparent = false,
  title,
  className,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const idPrefix = useId().replace(/:/g, "");
  const { isLive, pose } = usePose({
    status,
    animated,
    trackPointer,
    poseOffset,
    svgRef,
  });
  const { expression } = useExpression({
    status,
    isLive,
    override: expressionOverride,
  });

  return (
    <span
      data-testid="notion-avatar-host"
      style={{
        width: size,
        height: size,
        backgroundColor: transparent
          ? undefined
          : findNotionColor(config.color).disc,
      }}
      className="inline-flex shrink-0 items-center justify-center overflow-hidden rounded-full"
    >
      <NotionAvatarSvg
        svgRef={svgRef}
        config={config}
        status={status}
        expression={expression}
        pose={pose}
        isLive={isLive}
        els={motion}
        idPrefix={idPrefix}
        size={size}
        showBadge={showBadge}
        transparent
        title={title}
        className={className}
      />
    </span>
  );
}
