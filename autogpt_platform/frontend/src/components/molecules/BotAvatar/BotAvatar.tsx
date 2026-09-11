"use client";

import { motion } from "framer-motion";
import { useId, useRef } from "react";
import { BotAvatarSvg } from "./BotAvatarSvg";
import type { ExpressionId } from "./expressions";
import type { AvatarConfig, AvatarStatus } from "./helpers";
import type { Pose } from "./projection";
import { useExpression } from "./useExpression";
import { usePose } from "./usePose";

interface Props {
  config: AvatarConfig;
  status?: AvatarStatus;
  expression?: ExpressionId;
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
  expression: expressionOverride,
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
  const idPrefix = useId().replace(/:/g, "");
  const { isLive, pose } = usePose({
    status,
    animated,
    trackPointer,
    poseOffset,
    svgRef,
  });
  const { expression, isBlinking } = useExpression({
    status,
    isLive,
    override: expressionOverride,
  });

  return (
    <BotAvatarSvg
      svgRef={svgRef}
      config={config}
      status={status}
      expression={expression}
      pose={pose}
      isLive={isLive}
      isBlinking={isBlinking}
      els={motion}
      idPrefix={idPrefix}
      size={size}
      outline={outline}
      showBadge={showBadge}
      title={title}
      className={className}
    />
  );
}
