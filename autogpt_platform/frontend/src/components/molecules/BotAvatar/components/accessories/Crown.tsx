import { fromSurface, type Vec3 } from "../../parts";
import { edgeProps, GOLD, latAtY, ring } from "./geometry";
import { Disc, Slab, Strand } from "./primitives";
import type { AccessoryProps } from "./types";

const POINTS = 6;

export function Crown({
  anchors,
  body,
  pose,
  deep,
  outline,
  layer,
}: AccessoryProps) {
  const projection = { pose, body, layer };
  const edge = edgeProps(outline);
  const crownLat = latAtY(anchors.top + 21, anchors, body);
  const lons = Array.from(
    { length: POINTS },
    (_, index) => -Math.PI + (2 * Math.PI * index) / POINTS,
  );

  return (
    <g>
      {lons.map((lon, index) => {
        const spike: Vec3[] = [
          fromSurface(lon - 0.2, crownLat + 0.02, 1.04),
          fromSurface(lon, crownLat + 0.36, 1.2),
          fromSurface(lon + 0.2, crownLat + 0.02, 1.04),
        ];
        return (
          <g key={index}>
            <Slab
              {...projection}
              deep={deep}
              outline={outline}
              points={spike}
              fill={GOLD}
            />
          </g>
        );
      })}
      <Strand
        {...projection}
        deep={deep}
        points={ring(crownLat, 1.04, 40)}
        width={6.5}
        color={GOLD}
      />
      {lons.map((lon, index) => {
        const gem = fromSurface(lon + Math.PI / POINTS, crownLat, 1.06);
        return (
          <Disc
            {...projection}
            key={index}
            center={gem}
            normal={gem}
            minDepth={0.4}
          >
            <circle r={2.2} fill={deep} {...edge} />
          </Disc>
        );
      })}
    </g>
  );
}
