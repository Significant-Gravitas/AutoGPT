import { fromSurface, type Vec3 } from "../../parts";
import { edgeProps, eyeAnchors, HALF_PI } from "./geometry";
import { Disc, Strand } from "./primitives";
import type { AccessoryProps } from "./types";

function archBand(lat: number, radius: number, forward: number): Vec3[] {
  return Array.from({ length: 25 }, (_, index) => {
    const theta = Math.PI - lat - ((Math.PI - 2 * lat) * index) / 24;
    return [Math.cos(theta) * radius, Math.sin(theta) * radius, forward];
  });
}

export function Headset({
  anchors,
  body,
  pose,
  deep,
  outline,
  layer,
}: AccessoryProps) {
  const { eyeLat } = eyeAnchors(anchors, body);
  const projection = { pose, body, layer };
  const edge = edgeProps(outline);
  const bandLat = eyeLat + 0.14;
  const pads: Vec3[] = [-1, 1].map((side) =>
    fromSurface(side * HALF_PI, bandLat - 0.06, 1.03),
  );
  const boom: Vec3[] = [
    [1.02, Math.sin(bandLat) - 0.24, 0.2],
    [0.92, -0.3, 0.6],
    [0.5, -0.46, 0.98],
    [0.24, -0.48, 1.08],
  ];

  return (
    <g>
      <Strand
        {...projection}
        deep={deep}
        points={archBand(bandLat, 1.09, 0.06)}
        width={4.4}
      />
      <Strand
        {...projection}
        deep={deep}
        points={archBand(bandLat + 0.06, 1.05, 0.06)}
        width={2}
        color="#fff"
        opacity={0.35}
      />
      <Strand {...projection} deep={deep} points={boom} width={2.6} />
      <Disc {...projection} center={boom[3]} normal={[0, 0, 1]} minDepth={0.35}>
        <circle r={3.2} fill={deep} {...edge} />
      </Disc>
      {pads.map((pad, index) => (
        <Disc
          {...projection}
          key={index}
          center={pad}
          normal={[index === 0 ? -1 : 1, 0, 0]}
          minDepth={0.4}
          hideBelow={-0.25}
        >
          <rect
            x={-7}
            y={-10}
            width={14}
            height={20}
            rx={6}
            fill={deep}
            {...edge}
          />
          <rect
            x={-3.6}
            y={-6.4}
            width={7.2}
            height={12.8}
            rx={3.6}
            fill="#fff"
            opacity={0.3}
          />
        </Disc>
      ))}
    </g>
  );
}
