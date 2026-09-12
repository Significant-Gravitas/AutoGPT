import { fromSurface } from "../../parts";
import { GOLD } from "./geometry";
import { Disc } from "./primitives";
import type { AccessoryProps } from "./types";

const PETALS = [0, 72, 144, 216, 288];

export function Flower({ body, pose, deep, layer }: AccessoryProps) {
  const center = fromSurface(-0.84, 0.74, 1.04);
  return (
    <Disc
      pose={pose}
      body={body}
      layer={layer}
      center={center}
      normal={center}
      minDepth={0.66}
    >
      <path
        d="M2,4 C7,7 11,12 12,17"
        fill="none"
        stroke={deep}
        strokeWidth={2.2}
        strokeLinecap="round"
      />
      {PETALS.map((angle) => (
        <path
          key={angle}
          d="M0,-2.6 C5,-3.4 7.6,-8 5.4,-11.6 C3.6,-14.4 -3.6,-14.4 -5.4,-11.6 C-7.6,-8 -5,-3.4 0,-2.6 Z"
          fill="#fff"
          stroke={deep}
          strokeWidth={1.6}
          strokeLinejoin="round"
          transform={`rotate(${angle})`}
        />
      ))}
      <circle r={4} fill={GOLD} stroke={deep} strokeWidth={1.6} />
    </Disc>
  );
}
