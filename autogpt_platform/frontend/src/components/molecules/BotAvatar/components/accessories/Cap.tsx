import { fromSurface, type Vec3 } from "../../parts";
import { edgeProps, headUnit, latAtY, ring } from "./geometry";
import { Dome, Slab, Strand, Upright } from "./primitives";
import type { AccessoryProps } from "./types";

const VISOR_LONS = [-0.62, -0.42, -0.21, 0, 0.21, 0.42, 0.62];

export function Cap({
  anchors,
  body,
  pose,
  deep,
  outline,
  layer,
}: AccessoryProps) {
  const projection = { pose, body, layer };
  const edge = edgeProps(outline);
  const capLat = latAtY(anchors.eyeY - 22, anchors, body);

  const inner = VISOR_LONS.map((lon) => fromSurface(lon, capLat, 1.02));
  const outer: Vec3[] = VISOR_LONS.map((lon) => {
    const dip = 0.34 * (1 - (lon / 0.64) ** 2);
    const point = fromSurface(lon, capLat - dip, 1.0);
    return [point[0] * 1.03, point[1], point[2] + 0.55];
  });

  return (
    <g>
      <Slab
        {...projection}
        deep={deep}
        outline={outline}
        points={[...inner, ...outer.slice().reverse()]}
        strokeWidth={outline ? 2 : 1}
      />
      <Dome
        {...projection}
        lat={capLat}
        radius={1.03}
        fill={deep}
        outline={outline}
      />
      <Strand
        {...projection}
        deep={deep}
        points={ring(capLat, 1.035, 44)}
        width={3}
      />
      <Upright {...projection} height={1.04} unit={headUnit(body)}>
        <circle
          cy={-2}
          r={2.8}
          fill="#fff"
          stroke={deep}
          strokeWidth={1.8}
          {...edge}
        />
      </Upright>
    </g>
  );
}
