import { brimSquash, edgeProps, latAtY, ring } from "./geometry";
import { Dome, Strand, Upright } from "./primitives";
import type { AccessoryProps } from "./types";

export function Beanie({
  anchors,
  body,
  pose,
  deep,
  outline,
  layer,
}: AccessoryProps) {
  const projection = { pose, body, layer };
  const edge = edgeProps(outline);
  const capLat = latAtY(anchors.eyeY - 13, anchors, body);

  return (
    <g>
      <Dome
        {...projection}
        lat={capLat}
        radius={1.02}
        fill={deep}
        outline={outline}
      />
      {[0.16, 0.3, 0.44].map((offset) => (
        <Strand
          {...projection}
          key={offset}
          deep={deep}
          points={ring(capLat + offset, 1.04, 40)}
          width={1.6}
          color="#fff"
          opacity={0.25}
        />
      ))}
      <Strand
        {...projection}
        deep={deep}
        points={ring(capLat, 1.045, 44)}
        width={9}
      />
      <Strand
        {...projection}
        deep={deep}
        points={ring(capLat + 0.02, 1.055, 44)}
        width={2}
        color="#fff"
        opacity={0.3}
      />
      <Upright {...projection} height={1.03}>
        <circle
          cy={-brimSquash(pose, 3.5, 3)}
          r={5.5}
          fill="#fff"
          stroke={deep}
          strokeWidth={2}
          {...edge}
        />
      </Upright>
    </g>
  );
}
