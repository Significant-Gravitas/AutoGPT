import { INK, type AccessoryId, type ShapeAnchors } from "../helpers";
import {
  arcPath,
  ellipsoidFor,
  foreshortenTransform,
  coronalArc,
  isVisible,
  latitudeRing,
  project,
  surfacePointAt,
  type Pose,
  type Projected,
} from "../projection";

interface Props {
  accessory: AccessoryId;
  anchors: ShapeAnchors;
  pose: Pose;
  deep: string;
  outline: boolean;
}

const STAR =
  "M0,-5.5 L1.6,-1.7 L5.6,-1.6 L2.4,0.9 L3.5,4.8 L0,2.5 L-3.5,4.8 L-2.4,0.9 L-5.6,-1.6 L-1.6,-1.7 Z";
const HALF_PI = Math.PI / 2;

function Attached({
  at,
  minDepth = 0.1,
  children,
}: {
  at: Projected;
  minDepth?: number;
  children: React.ReactNode;
}) {
  if (!isVisible(at)) return null;
  return <g transform={foreshortenTransform(at, minDepth)}>{children}</g>;
}

export function Accessory({ accessory, anchors, pose, deep, outline }: Props) {
  const { cx, eyeY, eyeGap, top, bottom, width } = anchors;
  const body = ellipsoidFor(anchors);
  const left = cx - width / 2;
  const right = cx + width / 2;
  const edge = outline ? { stroke: INK, strokeWidth: 2 } : {};

  function at(x: number, y: number) {
    return project(surfacePointAt(x, y, body), pose, body);
  }

  switch (accessory) {
    case "glasses": {
      const eyeLat = surfacePointAt(cx, eyeY, body).lat;
      const lenses = [-1, 1].map((side) => at(cx + side * (eyeGap / 2), eyeY));
      const ears = [-1, 1].map((side) =>
        project({ lon: side * HALF_PI, lat: eyeLat }, pose, body),
      );
      return (
        <g fill="none" stroke={deep} strokeWidth={2.4} strokeLinecap="round">
          {lenses.every((lens) => lens.depth > 0.25) ? (
            <path
              d={`M${lenses[0].x + 9},${lenses[0].y} L${lenses[1].x - 9},${lenses[1].y}`}
            />
          ) : null}
          {lenses.map((lens, index) => {
            const ear = ears[index];
            if (lens.depth < 0.3 || ear.depth < 0.05) return null;
            const dx = ear.x - lens.x;
            const dy = ear.y - lens.y;
            const length = Math.hypot(dx, dy) || 1;
            const reach = Math.min(length, 9.5 * lens.depth + 8);
            const startX = lens.x + (dx / length) * 9.5 * lens.depth;
            const startY = lens.y + (dy / length) * 9.5;
            return (
              <path
                key={index}
                d={`M${startX},${startY} L${lens.x + (dx / length) * reach},${lens.y + (dy / length) * reach}`}
              />
            );
          })}
          {lenses.map((lens, index) => (
            <Attached key={index} at={lens} minDepth={0.15}>
              <circle r={9.5} />
            </Attached>
          ))}
        </g>
      );
    }
    case "headset": {
      const earLat = surfacePointAt(cx, eyeY - 2, body).lat;
      const ears = [-1, 1].map((side) => ({
        lon: side * (HALF_PI - 0.12),
        lat: earLat,
      }));
      const band = coronalArc(earLat + 0.2, 40, 0.16);
      const pads = ears.map((ear) => project(ear, pose, body));
      const mouthSide = at(cx + 9, eyeY + 16);
      const nearPad = pads[1].depth >= pads[0].depth ? 1 : 0;
      return (
        <g>
          <path
            d={arcPath(band, pose, body, -0.02)}
            fill="none"
            stroke={deep}
            strokeWidth={3.4}
            strokeLinecap="round"
          />
          {pads[nearPad].depth > 0.05 && isVisible(mouthSide) ? (
            <path
              d={`M${pads[nearPad].x},${pads[nearPad].y + 7} Q${pads[nearPad].x + (mouthSide.x - pads[nearPad].x) * 0.15},${mouthSide.y + 6} ${mouthSide.x},${mouthSide.y}`}
              fill="none"
              stroke={deep}
              strokeWidth={2.4}
              strokeLinecap="round"
            />
          ) : null}
          {isVisible(mouthSide) ? (
            <circle cx={mouthSide.x} cy={mouthSide.y} r={2.4} fill={deep} />
          ) : null}
          {pads.map((pad, index) =>
            pad.depth > -0.02 ? (
              <g key={index} transform={foreshortenTransform(pad, 0.45)}>
                <ellipse rx={6} ry={9} fill={deep} {...edge} />
                <ellipse rx={3} ry={5.5} fill="#fff" opacity={0.28} />
              </g>
            ) : null,
          )}
        </g>
      );
    }
    case "cap": {
      const capLat = surfacePointAt(cx, top + 11, body).lat;
      const crown = project({ lon: 0, lat: HALF_PI }, pose, body);
      const ring = latitudeRing(capLat, 40, -HALF_PI - 0.4, HALF_PI + 0.4)
        .map((point) => project(point, pose, body))
        .filter((point) => point.depth > 0);
      const brim = project({ lon: 0.95, lat: capLat + 0.02 }, pose, body);
      if (ring.length < 2) return null;
      const first = ring[0];
      const last = ring[ring.length - 1];
      const capTopY = Math.min(top - 2, crown.y - 4);
      const dome = `M${first.x},${first.y} ${ring
        .slice(1)
        .map((point) => `L${point.x},${point.y}`)
        .join(
          " ",
        )} C${last.x},${capTopY} ${first.x},${capTopY} ${first.x},${first.y} Z`;
      return (
        <g>
          <Attached at={brim} minDepth={0.15}>
            <rect
              x={-6}
              y={-3.5}
              width={40}
              height={7}
              rx={3.5}
              fill={deep}
              {...edge}
            />
          </Attached>
          <path d={dome} fill={deep} {...edge} strokeLinejoin="round" />
          <circle cx={crown.x} cy={capTopY + 1} r={2.6} fill={deep} {...edge} />
        </g>
      );
    }
    case "pen":
      return (
        <Attached at={at(right - 18, top + 14)} minDepth={0.55}>
          <g transform="rotate(28)">
            <rect
              x={-3}
              y={-30}
              width={6}
              height={30}
              rx={1.5}
              fill={deep}
              {...edge}
            />
            <path
              d="M-3,0 L0,7 L3,0 Z"
              fill="#F4E3B4"
              {...edge}
              strokeLinejoin="round"
            />
            <rect
              x={-3}
              y={-30}
              width={6}
              height={5}
              rx={1}
              fill="#fff"
              {...edge}
            />
          </g>
        </Attached>
      );
    case "star":
      return (
        <Attached at={at(right - 15, bottom - 17)} minDepth={0.55}>
          <circle r={9} fill={deep} {...edge} />
          <path d={STAR} fill="#fff" />
        </Attached>
      );
    case "bow":
      return (
        <Attached at={at(left + 17, top + 13)} minDepth={0.55}>
          <g transform="rotate(-18)">
            <path
              d="M0,0 L-11,-6 L-11,6 Z"
              fill={deep}
              {...edge}
              strokeLinejoin="round"
            />
            <path
              d="M0,0 L11,-6 L11,6 Z"
              fill={deep}
              {...edge}
              strokeLinejoin="round"
            />
            <circle r={2.6} fill={deep} {...edge} />
          </g>
        </Attached>
      );
    case "badge":
      return (
        <Attached at={at(left + 19, bottom - 24)} minDepth={0.3}>
          <rect
            x={-9}
            y={-6}
            width={18}
            height={12}
            rx={2.5}
            fill="#fff"
            stroke={deep}
            strokeWidth={2}
          />
          <circle cx={-4} cy={0} r={2} fill={deep} />
          <path
            d="M0,-1.5 L5.5,-1.5 M0,1.5 L4,1.5"
            stroke={deep}
            strokeWidth={1.6}
            strokeLinecap="round"
          />
        </Attached>
      );
    default:
      return null;
  }
}
