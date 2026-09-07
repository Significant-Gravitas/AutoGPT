import { INK, type AccessoryId, type ShapeAnchors } from "../helpers";
import {
  centroidLayer,
  discProjection,
  discTransform,
  fromSurface,
  isInFront,
  polygonPath,
  rotate,
  samplePolyline,
  splitPolyline,
  toScreen,
  type Layer,
  type Vec3,
} from "../parts";
import { ellipsoidFor, surfacePointAt, type Pose } from "../projection";

interface Props {
  accessory: AccessoryId;
  anchors: ShapeAnchors;
  pose: Pose;
  deep: string;
  outline: boolean;
  layer: Layer;
}

const STAR =
  "M0,-5.5 L1.6,-1.7 L5.6,-1.6 L2.4,0.9 L3.5,4.8 L0,2.5 L-3.5,4.8 L-2.4,0.9 L-5.6,-1.6 L-1.6,-1.7 Z";
const HALF_PI = Math.PI / 2;

function scale([x, y, z]: Vec3, factor: number): Vec3 {
  return [x * factor, y * factor, z * factor];
}

export function Accessory({
  accessory,
  anchors,
  pose,
  deep,
  outline,
  layer,
}: Props) {
  const { cx, eyeY, eyeGap, top } = anchors;
  const body = ellipsoidFor(anchors);
  const edge = outline ? { stroke: INK, strokeWidth: 2 } : {};
  const eyeLat = surfacePointAt(cx, eyeY, body).lat;
  const eyeLon = surfacePointAt(cx + eyeGap / 2, eyeY, body).lon;

  function Disc({
    center,
    normal,
    minDepth = 0.12,
    hideBelow = -1,
    children,
  }: {
    center: Vec3;
    normal: Vec3;
    minDepth?: number;
    hideBelow?: number;
    children: React.ReactNode;
  }) {
    const rotated = rotate(center, pose);
    if (rotated[2] < hideBelow) return null;
    if ((isInFront(rotated) ? "front" : "back") !== layer) return null;
    const projected = discProjection(center, normal, pose, body);
    return <g transform={discTransform(projected, minDepth)}>{children}</g>;
  }

  function Strand({
    points,
    width,
    color = deep,
  }: {
    points: Vec3[];
    width: number;
    color?: string;
  }) {
    const path = splitPolyline(samplePolyline(points), pose, body)[layer];
    if (!path) return null;
    return (
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={width}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    );
  }

  function Slab({ points, fill = deep }: { points: Vec3[]; fill?: string }) {
    if (centroidLayer(points, pose) !== layer) return null;
    return (
      <path
        d={polygonPath(points, pose, body)}
        fill={fill}
        {...edge}
        strokeLinejoin="round"
      />
    );
  }

  switch (accessory) {
    case "glasses": {
      const lenses = [-1, 1].map((side) =>
        fromSurface(side * eyeLon, eyeLat, 1.02),
      );
      const bridgeVisible = lenses.every((lens) =>
        isInFront(rotate(lens, pose)),
      );
      return (
        <g fill="none" stroke={deep} strokeWidth={2.4} strokeLinecap="round">
          {layer === "front" && bridgeVisible ? (
            <Strand
              points={[
                fromSurface(-eyeLon * 0.45, eyeLat, 1.03),
                fromSurface(eyeLon * 0.45, eyeLat, 1.03),
              ]}
              width={2.4}
            />
          ) : null}
          {[-1, 1].map((side) => (
            <Strand
              key={side}
              points={[
                fromSurface(side * (eyeLon + 0.32), eyeLat, 1.03),
                fromSurface(side * HALF_PI, eyeLat + 0.03, 1.02),
              ]}
              width={2.4}
            />
          ))}
          {lenses.map((lens, index) => (
            <Disc
              key={index}
              center={lens}
              normal={scale(lens, 1)}
              minDepth={0.15}
            >
              <circle r={9.5} />
            </Disc>
          ))}
        </g>
      );
    }
    case "headset": {
      const bandLat = eyeLat + 0.12;
      const band: Vec3[] = Array.from({ length: 25 }, (_, index) => {
        const theta =
          Math.PI - bandLat - ((Math.PI - 2 * bandLat) * index) / 24;
        return [Math.cos(theta) * 1.05, Math.sin(theta) * 1.05, 0.1];
      });
      const pads = [-1, 1].map((side) =>
        fromSurface(side * HALF_PI, bandLat - 0.02, 1.03),
      );
      const mic: Vec3[] = [
        [1.0, Math.sin(bandLat) - 0.15, 0.25],
        [0.85, -0.25, 0.75],
        [0.3, -0.42, 1.05],
      ];
      return (
        <g>
          <Strand points={band} width={3.4} />
          <Strand points={mic} width={2.4} />
          <Disc center={mic[2]} normal={[0, 0, 1]}>
            <circle r={2.6} fill={deep} />
          </Disc>
          {pads.map((pad, index) => (
            <Disc
              key={index}
              center={pad}
              normal={[index === 0 ? -1 : 1, 0, 0]}
              minDepth={0.45}
              hideBelow={-0.2}
            >
              <ellipse rx={6} ry={9} fill={deep} {...edge} />
              <ellipse rx={3} ry={5.5} fill="#fff" opacity={0.28} />
            </Disc>
          ))}
        </g>
      );
    }
    case "cap": {
      const capLat = surfacePointAt(cx, top + 19, body).lat;
      const capRadius = 1.12;
      const ring: Vec3[] = Array.from({ length: 49 }, (_, index) =>
        fromSurface(-Math.PI + (2 * Math.PI * index) / 48, capLat, capRadius),
      );
      const front = ring
        .map((point) => rotate(point, pose))
        .filter((point) => point[2] >= 0.02)
        .map((point) => toScreen(point, body));
      const brim: Vec3[] = [
        fromSurface(0.4, capLat - 0.04, 1.06),
        ...[0.4, 0.62, 0.84, 1.06, 1.28, 1.5].map((lon) =>
          fromSurface(lon, capLat - 0.2, 1.62),
        ),
        fromSurface(1.5, capLat - 0.04, 1.06),
      ];
      const dome =
        front.length > 2 && layer === "front"
          ? `M${front[0].x},${front[0].y} ${front
              .slice(1)
              .map((point) => `L${point.x},${point.y}`)
              .join(
                " ",
              )} A${body.rx * capRadius},${body.ry * capRadius} 0 0 0 ${front[0].x},${front[0].y} Z`
          : null;
      return (
        <g>
          <Slab points={brim} />
          {dome ? (
            <path d={dome} fill={deep} {...edge} strokeLinejoin="round" />
          ) : null}
          <Disc
            center={[0, capRadius + 0.02, 0]}
            normal={[0, 1, 0]}
            minDepth={0.5}
          >
            <circle r={2.6} fill={deep} {...edge} />
          </Disc>
        </g>
      );
    }
    case "pen": {
      const root = fromSurface(1.35, -0.12, 0.96);
      const tip = fromSurface(0.92, 1.0, 1.72);
      const along = (t: number): Vec3 => [
        root[0] + (tip[0] - root[0]) * t,
        root[1] + (tip[1] - root[1]) * t,
        root[2] + (tip[2] - root[2]) * t,
      ];
      return (
        <g>
          <Strand points={[root, along(0.84)]} width={7} />
          <Strand points={[along(0.7), along(0.78)]} width={7} color="#fff" />
          <Strand
            points={[along(0.84), along(0.95)]}
            width={5}
            color="#F4E3B4"
          />
          <Strand points={[along(0.95), tip]} width={2.6} color={INK} />
        </g>
      );
    }
    case "star": {
      const center = fromSurface(0.85, -0.7, 1.01);
      return (
        <Disc center={center} normal={center} minDepth={0.5}>
          <circle r={9} fill={deep} {...edge} />
          <path d={STAR} fill="#fff" />
        </Disc>
      );
    }
    case "bow": {
      const center = fromSurface(-0.95, 0.85, 1.02);
      return (
        <Disc center={center} normal={center} minDepth={0.5}>
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
        </Disc>
      );
    }
    case "badge": {
      const center = fromSurface(-0.55, -0.5, 1.01);
      return (
        <Disc center={center} normal={center} minDepth={0.3}>
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
        </Disc>
      );
    }
    default:
      return null;
  }
}
