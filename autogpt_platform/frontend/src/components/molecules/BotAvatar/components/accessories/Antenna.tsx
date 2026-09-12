import { edgeProps, headUnit } from "./geometry";
import { Upright } from "./primitives";
import type { AccessoryProps } from "./types";

export function Antenna({ body, pose, deep, outline, layer }: AccessoryProps) {
  const edge = edgeProps(outline);
  return (
    <Upright
      pose={pose}
      body={body}
      layer={layer}
      height={1.0}
      unit={headUnit(body)}
    >
      <path
        d="M0,1 C0,-4 -4,-8 -1.5,-12"
        fill="none"
        stroke={deep}
        strokeWidth={3.2}
        strokeLinecap="round"
      />
      <circle cx={-1.5} cy={-15.4} r={5} fill={deep} {...edge} />
      <circle cx={-3} cy={-16.8} r={1.5} fill="#fff" opacity={0.75} />
      <path
        d="M4.5,-19.5 L6.8,-21.6 M6.6,-14 L9.8,-14"
        fill="none"
        stroke={deep}
        strokeWidth={1.8}
        strokeLinecap="round"
        opacity={0.65}
      />
    </Upright>
  );
}
