import { BOW_LEFT, BOW_RIGHT, edgeProps, softProps } from "./geometry";

interface Props {
  deep: string;
  outline: boolean;
  tails?: boolean;
}

// The knot shared by the bow, the bow tie and the tied bands: two curved
// loops and a wrapped centre, drawn in the local frame of whatever pins it.
export function BowShape({ deep, outline, tails = false }: Props) {
  const soft = softProps(outline, deep);
  const edge = edgeProps(outline);
  return (
    <g>
      {tails ? (
        <path
          d="M-2,2 L-6,13 M2,2 L7,12"
          fill="none"
          stroke={deep}
          strokeWidth={3.4}
          strokeLinecap="round"
        />
      ) : null}
      <path d={BOW_LEFT} fill={deep} {...soft} strokeLinejoin="round" />
      <path d={BOW_RIGHT} fill={deep} {...soft} strokeLinejoin="round" />
      <rect
        x={-3.2}
        y={-3.6}
        width={6.4}
        height={7.2}
        rx={2.4}
        fill={deep}
        {...edge}
      />
      <path
        d="M-11,-4 Q-13,0 -11,4"
        fill="none"
        stroke="#fff"
        strokeWidth={1.6}
        strokeLinecap="round"
        opacity={0.4}
      />
    </g>
  );
}
