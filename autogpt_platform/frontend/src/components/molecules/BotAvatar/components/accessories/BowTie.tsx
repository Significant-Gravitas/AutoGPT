import { fromSurface } from "../../parts";
import { BowShape } from "./BowShape";
import { latAtY } from "./geometry";
import { Disc } from "./primitives";
import type { AccessoryProps } from "./types";

export function BowTie({
  anchors,
  body,
  pose,
  deep,
  outline,
  layer,
}: AccessoryProps) {
  const chin = fromSurface(0, latAtY(anchors.bottom - 14, anchors, body), 1.02);
  return (
    <Disc
      pose={pose}
      body={body}
      layer={layer}
      center={chin}
      normal={chin}
      minDepth={0.72}
    >
      <g transform="scale(1.3)">
        <BowShape deep={deep} outline={outline} />
      </g>
    </Disc>
  );
}
