import { motion } from "framer-motion";
import { INK, type AvatarStatus, type ShapeAnchors } from "../helpers";

interface Props {
  anchors: ShapeAnchors;
  status: AvatarStatus;
  blush: string;
  isLive: boolean;
  isBlinking: boolean;
  gaze: { x: number; y: number };
}

const PUPIL: Record<
  AvatarStatus,
  { rx: number; ry: number; dx: number; dy: number }
> = {
  idle: { rx: 4.5, ry: 6, dx: 0, dy: 0 },
  working: { rx: 4.5, ry: 4.2, dx: 1.5, dy: 1.6 },
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
  status,
  blush,
  isLive,
  isBlinking,
  gaze,
}: Props) {
  const { cx, eyeY, eyeGap } = anchors;
  const pupil = PUPIL[status];
  const isClosed = status === "done";
  const eyeScaleY = isBlinking ? 0.08 : 1;
  const eyes = [cx - eyeGap / 2, cx + eyeGap / 2];

  function popIn(from: { x?: number; y?: number; scale?: number }) {
    return isLive ? { ...from, opacity: 0 } : false;
  }

  return (
    <g data-status={status}>
      {eyes.map((eyeX, index) => (
        <motion.ellipse
          key={index}
          cx={eyeX + (index === 0 ? -6 : 6)}
          cy={eyeY + 8}
          rx={4.2}
          ry={2.4}
          fill={blush}
          initial={false}
          animate={{ opacity: status === "done" ? 0.85 : 0.5 }}
        />
      ))}
      {eyes.map((eyeX, index) => (
        <g key={index} transform={`translate(${eyeX} ${eyeY})`}>
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
              animate={{
                x: pupil.dx + gaze.x,
                y: pupil.dy + gaze.y,
                scaleY: eyeScaleY,
              }}
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
              d={index === 0 ? "M-5,-10 L5,-8" : "M-5,-8 L5,-10"}
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
      ))}
      <g transform={`translate(${cx} ${eyeY + 13})`}>
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
    </g>
  );
}
