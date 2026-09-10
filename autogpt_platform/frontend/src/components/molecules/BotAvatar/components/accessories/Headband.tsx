import { fromSurface } from "../../parts";
import { BowShape } from "./BowShape";
import { latAtY, ring } from "./geometry";
import { Disc, Strand } from "./primitives";
import type { AccessoryProps } from "./types";

export function Headband({
  anchors,
  body,
  pose,
  deep,
  outline,
  layer,
}: AccessoryProps) {
  const projection = { pose, body, layer };
  const bandLat = latAtY(anchors.eyeY - 16, anchors, body);
  const knot = fromSurface(-1.34, bandLat + 0.04, 1.06);

  return (
    <g>
      <Strand
        {...projection}
        deep={deep}
        points={ring(bandLat, 1.04, 48)}
        width={6}
      />
      <Strand
        {...projection}
        deep={deep}
        points={ring(bandLat, 1.05, 48)}
        width={2}
        color="#fff"
      />
      <Disc
        {...projection}
        center={knot}
        normal={knot}
        minDepth={0.62}
        hideBelow={-0.2}
      >
        <g transform="rotate(-24) scale(0.85)">
          <BowShape deep={deep} outline={outline} tails />
        </g>
      </Disc>
    </g>
  );
}
