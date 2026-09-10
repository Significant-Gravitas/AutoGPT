import { fromSurface, type Vec3 } from "../../parts";
import { edgeProps, latAtY, ring } from "./geometry";
import { Disc, Strand } from "./primitives";
import type { AccessoryProps } from "./types";

const DOT_LONS = [-0.72, -0.24, 0.24, 0.72];

export function Bandana({
  anchors,
  body,
  pose,
  deep,
  outline,
  layer,
}: AccessoryProps) {
  const projection = { pose, body, layer };
  const edge = edgeProps(outline);
  const bandLat = latAtY(anchors.eyeY - 15, anchors, body);
  const knot = fromSurface(-1.36, bandLat - 0.06, 1.06);
  const tail: Vec3[] = [
    fromSurface(-1.36, bandLat - 0.06, 1.06),
    fromSurface(-1.42, bandLat - 0.34, 1.12),
    fromSurface(-1.3, bandLat - 0.62, 1.14),
  ];

  return (
    <g>
      <Strand
        {...projection}
        deep={deep}
        points={ring(bandLat, 1.04, 48)}
        width={11}
      />
      {DOT_LONS.map((lon) => {
        const dot = fromSurface(lon, bandLat, 1.06);
        return (
          <Disc
            {...projection}
            key={lon}
            center={dot}
            normal={dot}
            minDepth={0.35}
          >
            <circle r={1.9} fill="#fff" opacity={0.85} />
          </Disc>
        );
      })}
      <Strand {...projection} deep={deep} points={tail} width={6} />
      <Disc {...projection} center={knot} normal={knot} minDepth={0.6}>
        <circle r={4.4} fill={deep} {...edge} />
      </Disc>
    </g>
  );
}
