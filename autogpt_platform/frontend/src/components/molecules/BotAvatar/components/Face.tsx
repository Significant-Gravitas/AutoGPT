import { motion } from "framer-motion";
import { INK, type AvatarStatus, type ShapeAnchors } from "../helpers";
import {
  ellipsoidFor,
  foreshortenTransform,
  isVisible,
  project,
  surfacePointAt,
  type Pose,
} from "../projection";

interface Props {
  anchors: ShapeAnchors;
  pose: Pose;
  status: AvatarStatus;
  blush: string;
  isLive: boolean;
  isBlinking: boolean;
}

const PUPIL: Record<
  AvatarStatus,
  { rx: number; ry: number; dx: number; dy: number }
> = {
  idle: { rx: 4.5, ry: 6, dx: 0, dy: 0 },
  working: { rx: 4.5, ry: 4.2, dx: 0.6, dy: 1 },
  waiting: { rx: 5, ry: 7, dx: 0, dy: -0.5 },
  done: { rx: 4.5, ry: 6, dx: 0, dy: 0 },
};

const MOUTH: Record<AvatarStatus, string> = {
  idle: "M-4,0 Q0,3.5 4,0",
  working: "M-3.5,0.5 L3.5,0.5",
  waiting: "M-2.4,0 A2.4,2.4 0 1 0 2.4,0 A2.4,2.4 0 1 0 -2.4,0",
  done: "M-6,-0.5 Q0,6.5 6,-0.5",
};

const SPRING = { type: "spring", stiffness: 260, damping: 20 } as const;

export function Face({
  anchors,
  pose,
  status,
  blush,
  isLive,
  isBlinking,
}: Props) {
  const { cx, eyeY, eyeGap } = anchors;
  const body = ellipsoidFor(anchors);
  const pupil = PUPIL[status];
  const isClosed = status === "done";
  const eyeScaleY = isBlinking ? 0.08 : 1;
  const sides = [-1, 1] as const;

  function popIn(from: { x?: number; y?: number; scale?: number }) {
    return isLive ? { ...from, opacity: 0 } : false;
  }

  const mouth = project(surfacePointAt(cx, eyeY + 13, body), pose, body);

  return (
    <g data-status={status}>
      {sides.map((side) => {
        const cheek = project(
          surfacePointAt(cx + side * (eyeGap / 2 + 6), eyeY + 8, body),
          pose,
          body,
        );
        if (!isVisible(cheek)) return null;
        return (
          <motion.ellipse
            key={side}
            transform={foreshortenTransform(cheek)}
            rx={4.2}
            ry={2.4}
            fill={blush}
            initial={false}
            animate={{ opacity: status === "done" ? 0.85 : 0.5 }}
          />
        );
      })}
      {sides.map((side) => {
        const eye = project(
          surfacePointAt(cx + side * (eyeGap / 2), eyeY, body),
          pose,
          body,
        );
        if (!isVisible(eye)) return null;
        return (
          <g key={side} transform={foreshortenTransform(eye, 0.12)}>
            {isClosed ? (
              <motion.path
                d="M-5,1 Q0,-5 5,1"
                fill="none"
                stroke={INK}
                strokeWidth={2.6}
                strokeLinecap="round"
                initial={popIn({ scale: 0.6 })}
                animate={{ scale: 1, opacity: 1 }}
                transition={SPRING}
              />
            ) : (
              <motion.g
                initial={false}
                animate={{ x: pupil.dx, y: pupil.dy, scaleY: eyeScaleY }}
                transition={isBlinking ? { duration: 0.06 } : SPRING}
              >
                <motion.ellipse
                  fill={INK}
                  initial={false}
                  animate={{ rx: pupil.rx, ry: pupil.ry }}
                  transition={SPRING}
                />
                <circle cx={-1.4} cy={-2.2} r={1.5} fill="#fff" />
              </motion.g>
            )}
            {status === "waiting" ? (
              <motion.path
                d="M-5,-10 Q0,-13 5,-10"
                fill="none"
                stroke={INK}
                strokeWidth={2}
                strokeLinecap="round"
                initial={popIn({ y: 3 })}
                animate={{ y: 0, opacity: 1 }}
                transition={SPRING}
              />
            ) : null}
            {status === "working" ? (
              <motion.path
                d={side < 0 ? "M-5,-10 L5,-8" : "M-5,-8 L5,-10"}
                fill="none"
                stroke={INK}
                strokeWidth={2}
                strokeLinecap="round"
                initial={popIn({ y: -3 })}
                animate={{ y: 0, opacity: 1 }}
                transition={SPRING}
              />
            ) : null}
          </g>
        );
      })}
      {isVisible(mouth) ? (
        <g transform={foreshortenTransform(mouth, 0.2)}>
          <motion.path
            fill="none"
            stroke={INK}
            strokeWidth={2.2}
            strokeLinecap="round"
            strokeLinejoin="round"
            initial={false}
            animate={{ d: MOUTH[status] }}
            transition={SPRING}
          />
        </g>
      ) : null}
    </g>
  );
}
