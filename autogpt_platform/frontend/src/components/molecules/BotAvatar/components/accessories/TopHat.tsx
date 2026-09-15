import { brimSquash, edgeProps, headUnit } from "./geometry";
import { Upright } from "./primitives";
import type { AccessoryProps } from "./types";

export function TopHat({ body, pose, deep, outline, layer }: AccessoryProps) {
  const edge = edgeProps(outline);

  return (
    <Upright
      pose={pose}
      body={body}
      layer={layer}
      height={0.99}
      unit={headUnit(body)}
    >
      <ellipse
        cy={-16}
        rx={15}
        ry={brimSquash(pose, 2.2, 3)}
        fill={deep}
        {...edge}
      />
      <rect
        x={-15}
        y={-16}
        width={30}
        height={16}
        fill={deep}
        stroke={deep}
        strokeWidth={1}
      />
      <rect x={-15} y={-8} width={30} height={5} fill="#fff" opacity={0.85} />
      <ellipse
        cy={0}
        rx={27}
        ry={brimSquash(pose, 3.2, 5)}
        fill={deep}
        {...edge}
      />
      <path
        d="M-11,-13.5 L-11,-9"
        stroke="#fff"
        strokeWidth={2.2}
        strokeLinecap="round"
        opacity={0.3}
      />
    </Upright>
  );
}
