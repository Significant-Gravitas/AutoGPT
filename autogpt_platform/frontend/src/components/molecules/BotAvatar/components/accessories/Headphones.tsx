import { fromSurface, type Vec3 } from "../../parts";
import { edgeProps, eyeAnchors, HALF_PI } from "./geometry";
import { Disc, Strand } from "./primitives";
import type { AccessoryProps } from "./types";

function archBand(lat: number, radius: number): Vec3[] {
  return Array.from({ length: 25 }, (_, index) => {
    const theta = Math.PI - lat - ((Math.PI - 2 * lat) * index) / 24;
    return [Math.cos(theta) * radius, Math.sin(theta) * radius, 0.04];
  });
}

export function Headphones({
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
  const bandLat = eyeLat + 0.18;
  const cups: Vec3[] = [-1, 1].map((side) =>
    fromSurface(side * HALF_PI, bandLat - 0.16, 1.05),
  );

  return (
    <g>
      <Strand
        {...projection}
        deep={deep}
        points={archBand(bandLat, 1.12)}
        width={6}
      />
      <Strand
        {...projection}
        deep={deep}
        points={archBand(bandLat + 0.16, 1.07)}
        width={3}
        color="#fff"
        opacity={0.35}
      />
      {[-1, 1].map((side) => (
        <Strand
          {...projection}
          key={side}
          deep={deep}
          points={[
            fromSurface(side * (HALF_PI - 0.04), bandLat + 0.08, 1.12),
            fromSurface(side * (HALF_PI - 0.02), bandLat - 0.1, 1.07),
          ]}
          width={3.4}
        />
      ))}
      {cups.map((cup, index) => (
        <Disc
          {...projection}
          key={index}
          center={cup}
          normal={[index === 0 ? -1 : 1, 0, 0]}
          minDepth={0.42}
          hideBelow={-0.3}
        >
          <rect
            x={-8.5}
            y={-12}
            width={17}
            height={24}
            rx={8}
            fill={deep}
            {...edge}
          />
          <ellipse rx={5} ry={8} fill="#fff" opacity={0.32} />
        </Disc>
      ))}
    </g>
  );
}
