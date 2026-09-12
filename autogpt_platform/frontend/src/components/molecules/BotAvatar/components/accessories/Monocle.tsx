import { fromSurface } from "../../parts";
import { eyeAnchors, latAtY } from "./geometry";
import { Disc, Strand } from "./primitives";
import type { AccessoryProps } from "./types";

export function Monocle({ anchors, body, pose, deep, layer }: AccessoryProps) {
  const { eyeLat, eyeLon } = eyeAnchors(anchors, body);
  const projection = { pose, body, layer };
  const lens = fromSurface(eyeLon, eyeLat, 1.02);
  const chinLat = latAtY(anchors.bottom - 12, anchors, body);
  const chain = [
    fromSurface(eyeLon * 1.05, eyeLat - 0.22, 1.04),
    fromSurface(eyeLon * 1.2, (eyeLat + chinLat) / 2, 1.05),
    fromSurface(eyeLon * 0.9, chinLat + 0.06, 1.04),
  ];

  return (
    <g>
      <Strand {...projection} deep={deep} points={chain} width={1.8} />
      <Disc {...projection} center={lens} normal={lens} minDepth={0.2}>
        <circle
          r={10}
          fill="#fff"
          fillOpacity={0.2}
          stroke={deep}
          strokeWidth={3}
        />
        <path
          d="M-6.8,-4.8 A10,10 0 0 1 -1.8,-8.8"
          fill="none"
          stroke="#fff"
          strokeWidth={1.8}
          strokeLinecap="round"
          opacity={0.7}
        />
      </Disc>
      <Disc {...projection} center={chain[2]} normal={chain[2]} minDepth={0.35}>
        <circle r={2.6} fill={deep} />
      </Disc>
    </g>
  );
}
