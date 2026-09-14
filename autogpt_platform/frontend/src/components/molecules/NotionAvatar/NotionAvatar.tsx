"use client";

import { motion } from "framer-motion";
import { useId, useRef } from "react";
import type { AvatarStatus, Expression } from "./expressions";
import type { NotionAvatarConfig } from "./helpers";
import { NotionAvatarSvg } from "./NotionAvatarSvg";
import type { Pose } from "./pose";
import { useExpression } from "./useExpression";
import { usePose } from "./usePose";

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
      transparent={transparent}
      title={title}
      className={className}
    />
  );
}
