import { fromSurface } from "../../parts";
import { Disc } from "./primitives";
import type { AccessoryProps } from "./types";

export function Badge({ body, pose, deep, layer }: AccessoryProps) {
  const center = fromSurface(-0.62, -0.46, 1.02);
  return (
    <Disc
      pose={pose}
      body={body}
      layer={layer}
      center={center}
      normal={center}
      minDepth={0.6}
    >
      <rect
        x={-2.6}
        y={-11}
        width={5.2}
        height={4}
        rx={1.4}
        fill={deep}
        stroke={deep}
        strokeWidth={1.6}
      />
      <rect
        x={-10}
        y={-8}
        width={20}
        height={15}
        rx={3}
        fill="#fff"
        stroke={deep}
        strokeWidth={2.2}
      />
      <rect x={-7.4} y={-5} width={7} height={7} rx={1.6} fill={deep} />
      <path
        d="M1.4,-3.4 L7,-3.4 M1.4,0 L7,0 M-7.4,4 L7,4"
        stroke={deep}
        strokeWidth={1.6}
        strokeLinecap="round"
      />
    </Disc>
  );
}
