import { fromSurface } from "../../parts";
import { edgeProps, eyeAnchors, GOLD, HALF_PI } from "./geometry";
import { Disc } from "./primitives";
import type { AccessoryProps } from "./types";

export function Earrings({
  anchors,
  body,
  pose,
  deep,
  outline,
  layer,
}: AccessoryProps) {
  const { eyeLat } = eyeAnchors(anchors, body);
  const projection = { pose, body, layer };

  return (
    <g>
      {[-1, 1].map((side) => {
        const stud = fromSurface(side * (HALF_PI - 0.3), eyeLat - 0.16, 1.02);
        return (
          <Disc
            {...projection}
            key={side}
            center={stud}
            normal={[side, 0, 0]}
            minDepth={0.8}
            hideBelow={-0.35}
          >
            <circle r={3} fill={GOLD} stroke={deep} strokeWidth={1.6} />
            <path
              d="M0,3 L0,7"
              stroke={deep}
              strokeWidth={1.6}
              strokeLinecap="round"
            />
            <path
              d="M0,7 C4,10 4,15 0,17 C-4,15 -4,10 0,7 Z"
              fill={GOLD}
              stroke={deep}
              strokeWidth={1.6}
              strokeLinejoin="round"
              {...edgeProps(outline)}
            />
          </Disc>
        );
      })}
    </g>
  );
}
