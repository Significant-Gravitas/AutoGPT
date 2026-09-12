import { fromSurface, type Vec3 } from "../../parts";
import { Slab } from "./primitives";
import type { AccessoryProps } from "./types";

function outerEar(side: number): Vec3[] {
  return [
    fromSurface(side * 1.08, 0.42, 1.0),
    fromSurface(side * 1.0, 0.74, 1.2),
    fromSurface(side * 0.86, 1.02, 1.42),
    fromSurface(side * 0.66, 0.94, 1.24),
    fromSurface(side * 0.44, 0.76, 1.04),
    fromSurface(side * 0.72, 0.5, 1.0),
  ];
}

function innerEar(side: number): Vec3[] {
  return [
    fromSurface(side * 0.94, 0.56, 1.02),
    fromSurface(side * 0.85, 0.9, 1.24),
    fromSurface(side * 0.68, 0.82, 1.08),
  ];
}

export function Ears({ body, pose, deep, outline, layer }: AccessoryProps) {
  const projection = { pose, body, layer };
  return (
    <g>
      {[-1, 1].map((side) => (
        <g key={side}>
          <Slab
            {...projection}
            deep={deep}
            outline={outline}
            points={outerEar(side)}
          />
          <Slab
            {...projection}
            deep={deep}
            outline={outline}
            points={innerEar(side)}
            fill="#fff"
            strokeWidth={outline ? 1.4 : 1}
          />
        </g>
      ))}
    </g>
  );
}
