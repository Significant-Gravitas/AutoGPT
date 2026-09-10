import { GOLD, headUnit } from "./geometry";
import { Disc } from "./primitives";
import type { AccessoryProps } from "./types";

export function Halo({ body, pose, layer }: AccessoryProps) {
  const unit = headUnit(body);
  return (
    <Disc
      pose={pose}
      body={body}
      layer={layer}
      center={[0, 1.3, 0]}
      normal={[0, 1, 0]}
      minDepth={0.2}
    >
      <circle
        r={25 * unit}
        fill="none"
        stroke={GOLD}
        strokeWidth={5.5 * unit}
      />
      <circle
        r={25 * unit}
        fill="none"
        stroke="#fff"
        strokeWidth={1.8 * unit}
        opacity={0.6}
      />
    </Disc>
  );
}
