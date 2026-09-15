import { fromSurface } from "../../parts";
import { GOLD, STAR_PATH } from "./geometry";
import { Disc } from "./primitives";
import type { AccessoryProps } from "./types";

export function Star({ body, pose, deep, layer }: AccessoryProps) {
  const center = fromSurface(0.88, 0.72, 1.03);
  const spark = fromSurface(0.52, 0.98, 1.06);
  const projection = { pose, body, layer };
  return (
    <g>
      <Disc {...projection} center={center} normal={center} minDepth={0.6}>
        <path
          d={STAR_PATH}
          fill={GOLD}
          stroke={deep}
          strokeWidth={2.2}
          strokeLinejoin="round"
        />
        <path d="M-1.6,-5 L0,-1 L-4,-1.4 Z" fill="#fff" opacity={0.6} />
      </Disc>
      <Disc {...projection} center={spark} normal={spark} minDepth={0.6}>
        <path
          d="M0,-5 Q0.8,-0.8 5,0 Q0.8,0.8 0,5 Q-0.8,0.8 -5,0 Q-0.8,-0.8 0,-5 Z"
          fill="#fff"
          stroke={deep}
          strokeWidth={1.2}
          strokeLinejoin="round"
        />
      </Disc>
    </g>
  );
}
