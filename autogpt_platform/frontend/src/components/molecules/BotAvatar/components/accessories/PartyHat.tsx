import { brimSquash, edgeProps, headUnit } from "./geometry";
import { Upright } from "./primitives";
import type { AccessoryProps } from "./types";

export function PartyHat({ body, pose, deep, outline, layer }: AccessoryProps) {
  const edge = edgeProps(outline);

  return (
    <Upright
      pose={pose}
      body={body}
      layer={layer}
      height={0.97}
      unit={headUnit(body)}
    >
      <path
        d="M0,-15 L13,2 L-13,2 Z"
        fill={deep}
        stroke={deep}
        strokeWidth={2}
        strokeLinejoin="round"
        {...edge}
      />
      <path
        d="M-8,-4.5 L8,-4.5 M-4.2,-10 L4.2,-10"
        fill="none"
        stroke="#fff"
        strokeWidth={2.6}
        strokeLinecap="round"
        opacity={0.85}
      />
      <ellipse
        cy={2}
        rx={13}
        ry={brimSquash(pose, 2.2, 3.4)}
        fill={deep}
        {...edge}
      />
      <circle cy={-17.4} r={3.6} fill="#fff" stroke={deep} strokeWidth={2} />
    </Upright>
  );
}
