import {
  findExpression,
  type BrowKind,
  type ExpressionId,
  type EyeSpec,
} from "../expressions";
import { INK, type AvatarStatus, type ShapeAnchors } from "../helpers";
import {
  ellipsoidFor,
  foreshortenTransform,
  isVisible,
  project,
  surfacePointAt,
  type Pose,
} from "../projection";
import type { SvgEls } from "../svgElements";

interface Props {
  anchors: ShapeAnchors;
  pose: Pose;
  status: AvatarStatus;
  expression: ExpressionId;
  blush: string;
  isLive: boolean;
  isBlinking: boolean;
  els: SvgEls;
}

const SPRING = { type: "spring", stiffness: 260, damping: 20 } as const;

const BROWS: Record<Exclude<BrowKind, "none">, (side: number) => string> = {
  raised: () => "M-5,-10 Q0,-13 5,-10",
  wave: () => "M-5,-10 Q-2.5,-13 0,-10 Q2.5,-7 5,-10",
  furrow: (side) => (side < 0 ? "M-5,-10 L5,-8" : "M-5,-8 L5,-10"),
  sad: (side) => (side < 0 ? "M-5,-8 L5,-10.5" : "M-5,-10.5 L5,-8"),
  flat: () => "M-5,-9.5 L5,-9.5",
};

function Eye({
  spec,
  isBlinking,
  isLive,
  els,
}: {
  spec: EyeSpec;
  isBlinking: boolean;
  isLive: boolean;
  els: SvgEls;
}) {
  const popIn = isLive ? { scale: 0.5, opacity: 0 } : false;
  if (spec.kind === "cross") {
    return (
      <els.path
        d="M-4,-4 L4,4 M4,-4 L-4,4"
        fill="none"
        stroke={INK}
        strokeWidth={2.4}
        strokeLinecap="round"
        initial={popIn}
        animate={{ scale: 1, opacity: 1 }}
        transition={SPRING}
      />
    );
  }
  if (spec.kind === "arc") {
    return (
      <els.path
        d="M-5,1 Q0,-5 5,1"
        fill="none"
        stroke={INK}
        strokeWidth={2.6}
        strokeLinecap="round"
        initial={popIn}
        animate={{ scale: 1, opacity: 1 }}
        transition={SPRING}
      />
    );
  }
  const closed = isBlinking || spec.kind === "flat";
  return (
    <els.g
      initial={false}
      animate={{
        x: spec.dx,
        y: spec.dy,
        rotate: spec.tilt,
        scaleY: isBlinking ? 0.08 : 1,
      }}
      transition={isBlinking ? { duration: 0.06 } : SPRING}
    >
      <els.ellipse
        fill={INK}
        initial={false}
        animate={{ rx: spec.rx, ry: spec.ry }}
        transition={SPRING}
      />
      {closed ? null : <circle cx={-1.4} cy={-2.2} r={1.5} fill="#fff" />}
    </els.g>
  );
}

export function Face({
  anchors,
  pose,
  status,
  expression,
  blush,
  isLive,
  isBlinking,
  els,
}: Props) {
  const { cx, eyeY, eyeGap } = anchors;
  const body = ellipsoidFor(anchors);
  const spec = findExpression(expression);
  const sides = [-1, 1] as const;
  const mouth = project(surfacePointAt(cx, eyeY + 13, body), pose, body);
  const browPath = spec.brow === "none" ? null : BROWS[spec.brow];
  const browWaves = spec.brow === "wave" && isLive;

  return (
    <g data-status={status} data-expression={expression}>
      {sides.map((side) => {
        const cheek = project(
          surfacePointAt(cx + side * (eyeGap / 2 + 6), eyeY + 8, body),
          pose,
          body,
        );
        if (!isVisible(cheek)) return null;
        return (
          <els.ellipse
            key={side}
            transform={foreshortenTransform(cheek)}
            rx={4.2}
            ry={2.4}
            fill={blush}
            initial={false}
            animate={{ opacity: spec.blush }}
          />
        );
      })}
      {sides.map((side) => {
        const point = project(
          surfacePointAt(cx + side * (eyeGap / 2), eyeY, body),
          pose,
          body,
        );
        if (!isVisible(point)) return null;
        const eyeSpec = side < 0 ? spec.left : spec.right;
        return (
          <g key={side} transform={foreshortenTransform(point, 0.12)}>
            <Eye
              spec={eyeSpec}
              isBlinking={isBlinking}
              isLive={isLive}
              els={els}
            />
            {browPath ? (
              <els.path
                d={browPath(side)}
                fill="none"
                stroke={INK}
                strokeWidth={2}
                strokeLinecap="round"
                initial={isLive ? { y: 3, opacity: 0 } : false}
                animate={
                  browWaves
                    ? { y: [0, -2, 0, 2, 0], opacity: 1 }
                    : { y: 0, opacity: 1 }
                }
                transition={
                  browWaves
                    ? { duration: 1.4, repeat: Infinity, ease: "easeInOut" }
                    : SPRING
                }
              />
            ) : null}
          </g>
        );
      })}
      {isVisible(mouth) ? (
        <g transform={foreshortenTransform(mouth, 0.2)}>
          <els.path
            fill="none"
            stroke={INK}
            strokeWidth={2.2}
            strokeLinecap="round"
            strokeLinejoin="round"
            initial={false}
            d={spec.mouth}
            transition={SPRING}
          />
        </g>
      ) : null}
    </g>
  );
}
