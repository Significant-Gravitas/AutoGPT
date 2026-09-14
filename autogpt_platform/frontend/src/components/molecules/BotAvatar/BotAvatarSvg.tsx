import { cn } from "@/lib/utils";
import type { RefObject } from "react";
import { Accessory } from "./components/Accessory";
import { Face } from "./components/Face";
import { StatusBadge } from "./components/StatusBadge";
import type { ExpressionId } from "./expressions";
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
import type { SvgEls } from "./svgElements";

interface Props {
  config: AvatarConfig;
  status: AvatarStatus;
  expression: ExpressionId;
  pose: Pose;
  isLive: boolean;
  isBlinking: boolean;
  els: SvgEls;
  /** Prefix for the gradient id — unique per instance on a page. */
  idPrefix: string;
  size?: number;
  outline?: boolean;
  showBadge?: boolean;
  title?: string;
  className?: string;
  svgRef?: RefObject<SVGSVGElement>;
}

// The avatar's markup with every moving part passed in: BotAvatar drives
// it from hooks in the browser; the /avatars/[file] route renders it once,
// at rest, with plain elements.
export function BotAvatarSvg({
  config,
  status,
  expression,
  pose,
  isLive,
  isBlinking,
  els,
  idPrefix,
  size = 96,
  outline = false,
  showBadge = true,
  title,
  className,
  svgRef,
}: Props) {
  const gradientId = `${idPrefix}-body-${config.color}`;
  const shape = findShape(config.shape);
  const color = findColor(config.color);
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
      data-expression={expression}
      data-outline={outline}
      className={cn("shrink-0 overflow-visible", className)}
    >
      <defs>
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
          expression={expression}
          blush={color.mid}
          isLive={isLive}
          isBlinking={isBlinking}
          els={els}
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
        <StatusBadge
          status={status}
          anchors={shape.anchors}
          isLive={isLive}
          els={els}
        />
      ) : null}
    </svg>
  );
}
