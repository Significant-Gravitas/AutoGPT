import type { AvatarStatus } from "../expressions";
import type { SvgEls } from "../svgElements";

interface Props {
  status: AvatarStatus;
  /** Side of the square viewBox the avatar is drawn in. */
  viewBox: number;
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

// The badge geometry below is drawn at a radius of 8; on the avatar's much
// larger canvas it rides on the disc's lower right, scaled to match.
const BADGE_SCALE = 7;
const DIAGONAL = Math.SQRT1_2;

export function StatusBadge({ status, viewBox, isLive, els }: Props) {
  if (status === "idle") return null;
  const centre = viewBox / 2;
  const x = centre + centre * DIAGONAL;
  const y = centre + centre * DIAGONAL;

  return (
    <g transform={`translate(${x} ${y}) scale(${BADGE_SCALE})`}>
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
