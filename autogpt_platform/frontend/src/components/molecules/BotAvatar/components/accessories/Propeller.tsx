import type { Vec3 } from "../../parts";
import { edgeProps, latAtY } from "./geometry";
import { Disc, Dome, Slab, Strand } from "./primitives";
import type { AccessoryProps } from "./types";

const HUB = 0.24;

export function Propeller({
  anchors,
  body,
  pose,
  deep,
  outline,
  layer,
}: AccessoryProps) {
  const projection = { pose, body, layer };
  const edge = edgeProps(outline);
  const capLat = latAtY(anchors.top + 18, anchors, body);
  const capRadius = 1.03;
  const top = capRadius + HUB;
  const spin = ((pose.bob + 10) * 0.9 + pose.yaw * 6) % (Math.PI * 2);
  const dir: Vec3 = [Math.cos(spin), 0, Math.sin(spin)];
  const side: Vec3 = [-Math.sin(spin), 0, Math.cos(spin)];

  function blade(sign: number): Vec3[] {
    const reach = 0.62 * sign;
    const width = 0.14;
    return [
      [dir[0] * 0.08 * sign, top, dir[2] * 0.08 * sign],
      [
        dir[0] * reach + side[0] * width,
        top + 0.05,
        dir[2] * reach + side[2] * width,
      ],
      [dir[0] * reach * 1.1, top, dir[2] * reach * 1.1],
      [
        dir[0] * reach - side[0] * width,
        top - 0.05,
        dir[2] * reach - side[2] * width,
      ],
    ];
  }

  return (
    <g>
      <Dome
        {...projection}
        lat={capLat}
        radius={capRadius}
        fill={deep}
        outline={outline}
      />
      <Strand
        {...projection}
        deep={deep}
        points={[
          [0, capRadius, 0],
          [0, top, 0],
        ]}
        width={3}
      />
      {[1, -1].map((sign) => (
        <Slab
          {...projection}
          key={sign}
          deep={deep}
          outline={outline}
          points={blade(sign)}
          strokeWidth={outline ? 1.6 : 1}
        />
      ))}
      <Disc {...projection} center={[0, top, 0]} normal={[0, 0, 1]}>
        <circle r={2.8} fill={deep} {...edge} />
      </Disc>
    </g>
  );
}
