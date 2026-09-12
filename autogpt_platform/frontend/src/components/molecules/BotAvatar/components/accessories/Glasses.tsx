import { fromSurface, type Vec3 } from "../../parts";
import { eyeAnchors, TEMPLE_LON } from "./geometry";
import { Disc, Strand } from "./primitives";
import type { AccessoryProps } from "./types";

export function Glasses({ anchors, body, pose, deep, layer }: AccessoryProps) {
  const { eyeLat, eyeLon } = eyeAnchors(anchors, body);
  const projection = { pose, body, layer };
  const lenses: Vec3[] = [-1, 1].map((side) =>
    fromSurface(side * eyeLon, eyeLat, 1.02),
  );
  const bridge: Vec3[] = [
    fromSurface(-eyeLon * 0.42, eyeLat + 0.04, 1.04),
    fromSurface(0, eyeLat + 0.08, 1.05),
    fromSurface(eyeLon * 0.42, eyeLat + 0.04, 1.04),
  ];

  return (
    <g>
      <Strand {...projection} deep={deep} points={bridge} width={2.6} />
      {[-1, 1].map((side) => (
        <Strand
          {...projection}
          deep={deep}
          key={side}
          points={[
            fromSurface(side * (eyeLon + 0.36), eyeLat + 0.05, 1.03),
            fromSurface(side * TEMPLE_LON, eyeLat + 0.06, 1.02),
          ]}
          width={2.6}
        />
      ))}
      {lenses.map((lens, index) => (
        <Disc
          {...projection}
          key={index}
          center={lens}
          normal={lens}
          minDepth={0.18}
        >
          <rect
            x={-10.5}
            y={-8}
            width={21}
            height={16}
            rx={4.5}
            fill="#fff"
            fillOpacity={0.18}
            stroke={deep}
            strokeWidth={2.8}
            strokeLinejoin="round"
          />
        </Disc>
      ))}
    </g>
  );
}
