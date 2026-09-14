import { cn } from "@/lib/utils";
import type { RefObject } from "react";
import { StatusBadge } from "./components/StatusBadge";
import {
  FRAME_OFFSET_Y,
  FRAME_SCALE,
  haloFilter,
  layerFill,
  layerMarkup,
} from "./compose";
import type { AvatarStatus, Expression } from "./expressions";
import {
  encodeNotionConfig,
  findNotionColor,
  VIEWBOX,
  type NotionAvatarConfig,
} from "./helpers";
import { featureTransform, headTransform, type Pose } from "./pose";
import type { NotionCategory } from "./metadata.generated";
import type { SvgEls } from "./svgElements";

// Draw order splits into three bands so the features that follow a head turn
// (eyes, brows and whatever sits on them) can ride in their own group without
// disturbing what stacks above or below them.
const BEHIND: NotionCategory[] = ["face", "nose", "mouth"];
const FEATURES: NotionCategory[] = ["eyes", "eyebrows", "glasses"];
const IN_FRONT: NotionCategory[] = ["hair", "accessories", "details", "beard"];

interface Props {
  config: NotionAvatarConfig;
  status: AvatarStatus;
  expression: Expression;
  pose: Pose;
  isLive: boolean;
  els: SvgEls;
  /** Prefix for ids inside the artwork — unique per instance on a page. */
  idPrefix: string;
  size?: number;
  showBadge?: boolean;
  transparent?: boolean;
  title?: string;
  className?: string;
  svgRef?: RefObject<SVGSVGElement>;
}

export function NotionAvatarSvg({
  config,
  status,
  expression,
  pose,
  isLive,
  els,
  idPrefix,
  size = 96,
  showBadge = true,
  transparent = false,
  title,
  className,
  svgRef,
}: Props) {
  const color = findNotionColor(config.color);
  const inset = (VIEWBOX / 2) * (1 - FRAME_SCALE);

  function partIndex(category: NotionCategory): number {
    if (category === "eyes" && expression.eyes !== undefined)
      return expression.eyes;
    if (category === "eyebrows" && expression.eyebrows !== undefined)
      return expression.eyebrows;
    if (category === "mouth" && expression.mouth !== undefined)
      return expression.mouth;
    return config.parts[category];
  }

  function band(categories: NotionCategory[]) {
    return categories.map((category) => {
      const index = partIndex(category);
      const fill = layerFill(category);
      return (
        <g
          key={category}
          data-layer={category}
          fill={fill}
          dangerouslySetInnerHTML={{
            __html: layerMarkup(category, index, idPrefix),
          }}
        />
      );
    });
  }

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${VIEWBOX} ${VIEWBOX}`}
      width={size}
      height={size}
      role="img"
      aria-label={title ?? `${color.label} avatar, ${status}`}
      data-testid="notion-avatar"
      data-avatar={encodeNotionConfig(config)}
      data-status={status}
      className={cn("shrink-0", className)}
    >
      {transparent ? null : (
        <circle
          cx={VIEWBOX / 2}
          cy={VIEWBOX / 2}
          r={VIEWBOX / 2}
          fill={color.disc}
        />
      )}
      <defs dangerouslySetInnerHTML={{ __html: haloFilter(idPrefix, true) }} />
      <g filter={`url(#${idPrefix}halo)`}>
        <g
          transform={`translate(${inset} ${inset + FRAME_OFFSET_Y}) scale(${FRAME_SCALE})`}
        >
          <g transform={headTransform(pose, VIEWBOX / 2)}>
            {band(BEHIND)}
            <g transform={featureTransform(pose)}>{band(FEATURES)}</g>
            {band(IN_FRONT)}
          </g>
        </g>
      </g>
      {showBadge ? (
        <StatusBadge
          status={status}
          viewBox={VIEWBOX}
          isLive={isLive}
          els={els}
        />
      ) : null}
    </svg>
  );
}
