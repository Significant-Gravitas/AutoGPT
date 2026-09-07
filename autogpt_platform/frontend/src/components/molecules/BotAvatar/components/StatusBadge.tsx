import { motion } from "framer-motion";
import type { AvatarStatus, ShapeAnchors } from "../helpers";

interface Props {
  status: AvatarStatus;
  anchors: ShapeAnchors;
  isLive: boolean;
}

const FILL: Record<Exclude<AvatarStatus, "idle">, string> = {
  working: "#3B6FD1",
  waiting: "#E8A317",
  done: "#22A05B",
};

export function StatusBadge({ status, anchors, isLive }: Props) {
  if (status === "idle") return null;
  const x = anchors.cx + anchors.width / 2 - 4;
  const y = anchors.bottom - 6;

  return (
    <g transform={`translate(${x} ${y})`}>
      <motion.g
        initial={isLive ? { scale: 0 } : false}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 400, damping: 18 }}
      >
        <circle r={8} fill={FILL[status]} stroke="#fff" strokeWidth={2.5} />
        {status === "working" ? (
          <motion.circle
            r={3}
            fill="#fff"
            initial={false}
            animate={
              isLive ? { scale: [1, 0.55, 1], opacity: [1, 0.6, 1] } : undefined
            }
            transition={{ duration: 1, repeat: Infinity, ease: "easeInOut" }}
          />
        ) : null}
        {status === "done" ? (
          <path
            d="M-3.5,0.2 L-1,2.8 L3.8,-2.6"
            fill="none"
            stroke="#fff"
            strokeWidth={2.2}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ) : null}
        {status === "waiting" ? (
          <path
            d="M0,-3.6 L0,0.8 M0,3.4 L0,3.5"
            fill="none"
            stroke="#fff"
            strokeWidth={2.4}
            strokeLinecap="round"
          />
        ) : null}
      </motion.g>
    </g>
  );
}
