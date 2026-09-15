import { fromSurface } from "../../parts";
import { BowShape } from "./BowShape";
import { Disc } from "./primitives";
import type { AccessoryProps } from "./types";

export function Bow({ body, pose, deep, outline, layer }: AccessoryProps) {
  const center = fromSurface(-0.82, 0.76, 1.04);
  return (
    <Disc
      pose={pose}
      body={body}
      layer={layer}
      center={center}
      normal={center}
      minDepth={0.68}
    >
      <g transform="rotate(-16)">
        <BowShape deep={deep} outline={outline} tails />
      </g>
    </Disc>
  );
}
