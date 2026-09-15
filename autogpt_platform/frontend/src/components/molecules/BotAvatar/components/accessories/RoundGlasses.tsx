import { fromSurface, type Vec3 } from "../../parts";
import { eyeAnchors, TEMPLE_LON } from "./geometry";
import { Disc, Strand } from "./primitives";
import type { AccessoryProps } from "./types";

export function RoundGlasses({
  anchors,
  body,
  pose,
  deep,
  layer,
}: AccessoryProps) {
  const { eyeLat, eyeLon } = eyeAnchors(anchors, body);
  const projection = { pose, body, layer };
  const lenses: Vec3[] = [-1, 1].map((side) =>
    fromSurface(side * eyeLon, eyeLat, 1.02),
  );

  return (
    <g>
      <Strand
        {...projection}
        deep={deep}
        points={[
          fromSurface(-eyeLon * 0.45, eyeLat + 0.02, 1.04),
          fromSurface(0, eyeLat + 0.09, 1.05),
          fromSurface(eyeLon * 0.45, eyeLat + 0.02, 1.04),
        ]}
        width={2.2}
      />
      {[-1, 1].map((side) => (
        <Strand
          {...projection}
          deep={deep}
          key={side}
          points={[
            fromSurface(side * (eyeLon + 0.34), eyeLat + 0.04, 1.03),
            fromSurface(side * TEMPLE_LON, eyeLat + 0.05, 1.02),
          ]}
          width={2.2}
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
          <circle
            r={9.5}
            fill="#fff"
            fillOpacity={0.18}
            stroke={deep}
            strokeWidth={2.6}
          />
          <path
            d="M-6.4,-4.6 A9.5,9.5 0 0 1 -1.6,-8.4"
            fill="none"
            stroke="#fff"
            strokeWidth={1.8}
            strokeLinecap="round"
            opacity={0.7}
          />
        </Disc>
      ))}
    </g>
  );
}
