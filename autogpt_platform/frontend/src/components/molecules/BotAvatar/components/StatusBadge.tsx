import type { AvatarStatus, ShapeAnchors } from "../helpers";
import type { SvgEls } from "../svgElements";

interface Props {
  status: AvatarStatus;
  anchors: ShapeAnchors;
  isLive: boolean;
  els: SvgEls;
}

const FILL: Record<Exclude<AvatarStatus, "idle">, string> = {
  thinking: "#7C3AED",
  working: "#3B6FD1",
  waiting: "#E8A317",
  done: "#22A05B",
  failed: "#DC2626",
  sleeping: "#71717A",
};

export function StatusBadge({ status, anchors, isLive, els }: Props) {
  if (status === "idle") return null;
  const x = anchors.cx + anchors.width / 2 - 4;
  const y = anchors.bottom - 6;

  return (
    <g transform={`translate(${x} ${y})`}>
      <els.g
        initial={isLive ? { scale: 0 } : false}
        animate={{ scale: 1 }}
        transition={{ type: "spring", stiffness: 400, damping: 18 }}
      >
        <circle r={8} fill={FILL[status]} stroke="#fff" strokeWidth={2.5} />
        {status === "working" ? (
          <els.circle
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
        {status === "thinking" ? (
          <g fill="#fff">
            {[-3.2, 0, 3.2].map((x, index) => (
              <els.circle
                key={x}
                cx={x}
                r={1.4}
                initial={false}
                animate={isLive ? { opacity: [0.35, 1, 0.35] } : undefined}
                transition={{
                  duration: 1.2,
                  repeat: Infinity,
                  delay: index * 0.2,
                  ease: "easeInOut",
                }}
              />
            ))}
          </g>
        ) : null}
        {status === "failed" ? (
          <path
            d="M-3,-3 L3,3 M3,-3 L-3,3"
            fill="none"
            stroke="#fff"
            strokeWidth={2.2}
            strokeLinecap="round"
          />
        ) : null}
        {status === "sleeping" ? (
          <path
            d="M-3,-3 L3,-3 L-3,3 L3,3"
            fill="none"
            stroke="#fff"
            strokeWidth={1.8}
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
      </els.g>
    </g>
  );
}
