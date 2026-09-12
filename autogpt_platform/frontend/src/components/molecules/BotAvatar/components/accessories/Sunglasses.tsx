import { INK } from "../../helpers";
import { fromSurface, type Vec3 } from "../../parts";
import { eyeAnchors, TEMPLE_LON } from "./geometry";
import { Disc, Strand } from "./primitives";
import type { AccessoryProps } from "./types";

export function Sunglasses({
  anchors,
  body,
  pose,
  deep,
  layer,
}: AccessoryProps) {
  const { eyeLat, eyeLon } = eyeAnchors(anchors, body);
  const projection = { pose, body, layer };
  const lenses: Vec3[] = [-1, 1].map((side) =>
    fromSurface(side * eyeLon, eyeLat - 0.02, 1.02),
  );
  const brow: Vec3[] = [-1, -0.5, 0, 0.5, 1].map((t) =>
    fromSurface(t * (eyeLon + 0.3), eyeLat + 0.12 - 0.02 * t * t, 1.04),
  );

  return (
    <g>
      {lenses.map((lens, index) => (
        <Disc
          {...projection}
          key={index}
          center={lens}
          normal={lens}
          minDepth={0.18}
        >
          <path
            d="M-11,-7 L11,-7 Q12,4 5,7.5 Q-3,9 -9,3 Q-11,0 -11,-7 Z"
            fill={INK}
            stroke={deep}
            strokeWidth={2}
            strokeLinejoin="round"
          />
          <path
            d="M-7.5,-4 L-2,-4 L-6,4 L-9.5,2 Z"
            fill="#fff"
            opacity={0.35}
          />
        </Disc>
      ))}
      <Strand {...projection} deep={deep} points={brow} width={3.2} />
      {[-1, 1].map((side) => (
        <Strand
          {...projection}
          deep={deep}
          key={side}
          points={[
            fromSurface(side * (eyeLon + 0.34), eyeLat + 0.08, 1.03),
            fromSurface(side * TEMPLE_LON, eyeLat + 0.06, 1.02),
          ]}
          width={2.6}
        />
      ))}
    </g>
  );
}
