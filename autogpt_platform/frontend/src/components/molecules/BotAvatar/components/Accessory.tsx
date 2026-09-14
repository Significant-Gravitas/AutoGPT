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
import {
  ellipsoidFor,
  surfacePointAt,
  type Ellipsoid,
  type Pose,
} from "../projection";

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

interface ProjectionProps {
  pose: Pose;
  body: Ellipsoid;
  layer: Layer;
}

interface DiscProps extends ProjectionProps {
  center: Vec3;
  normal: Vec3;
  minDepth?: number;
  hideBelow?: number;
  children: React.ReactNode;
}

function Disc({
  pose,
  body,
  layer,
  center,
  normal,
  minDepth = 0.12,
  hideBelow = -1,
  children,
}: DiscProps) {
  const rotated = rotate(center, pose);
  if (rotated[2] < hideBelow) return null;
  if ((isInFront(rotated) ? "front" : "back") !== layer) return null;
  const projected = discProjection(center, normal, pose, body);
  return <g transform={discTransform(projected, minDepth)}>{children}</g>;
}

interface StrandProps extends ProjectionProps {
  deep: string;
  points: Vec3[];
  width: number;
  color?: string;
}

function Strand({
  pose,
  body,
  layer,
  deep,
  points,
  width,
  color = deep,
}: StrandProps) {
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

interface SlabProps extends ProjectionProps {
  points: Vec3[];
  fill?: string;
  deep: string;
  outline: boolean;
}

function Slab({
  points,
  deep,
  fill = deep,
  pose,
  body,
  layer,
  outline,
}: SlabProps) {
  if (centroidLayer(points, pose) !== layer) return null;
  return (
    <path
      d={polygonPath(points, pose, body)}
      fill={fill}
      stroke={outline ? INK : fill}
      strokeWidth={outline ? 2 : 3.5}
      strokeLinejoin="round"
      strokeLinecap="round"
    />
  );
}

export function Accessory({
  accessory,
  anchors,
  pose,
  deep,
  outline,
  layer,
}: Props) {
  const { cx, eyeY, eyeGap, top, bottom } = anchors;
  const body = ellipsoidFor(anchors);
  const projection = { pose, body, layer };
  const edge = outline ? { stroke: INK, strokeWidth: 2 } : {};
  const soft = outline
    ? { stroke: INK, strokeWidth: 2 }
    : { stroke: deep, strokeWidth: 3 };
  const eyeLat = surfacePointAt(cx, eyeY, body).lat;
  const eyeLon = surfacePointAt(cx + eyeGap / 2, eyeY, body).lon;

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
              {...projection}
              deep={deep}
              points={[
                fromSurface(-eyeLon * 0.45, eyeLat, 1.03),
                fromSurface(eyeLon * 0.45, eyeLat, 1.03),
              ]}
              width={2.4}
            />
          ) : null}
          {[-1, 1].map((side) => (
            <Strand
              {...projection}
              deep={deep}
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
              {...projection}
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
          <Strand {...projection} deep={deep} points={band} width={3.4} />
          <Strand {...projection} deep={deep} points={mic} width={2.4} />
          <Disc {...projection} center={mic[2]} normal={[0, 0, 1]}>
            <circle r={2.6} fill={deep} />
          </Disc>
          {pads.map((pad, index) => (
            <Disc
              {...projection}
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
    case "star": {
      const center = fromSurface(0.85, -0.7, 1.01);
      return (
        <Disc {...projection} center={center} normal={center} minDepth={0.5}>
          <circle r={9} fill={deep} {...edge} />
          <path
            d={STAR}
            fill="#fff"
            stroke="#fff"
            strokeWidth={1.6}
            strokeLinejoin="round"
          />
        </Disc>
      );
    }
    case "bow": {
      const center = fromSurface(-0.95, 0.85, 1.02);
      return (
        <Disc {...projection} center={center} normal={center} minDepth={0.5}>
          <g transform="rotate(-18)">
            <path
              d="M0,0 L-11,-6 L-11,6 Z"
              fill={deep}
              {...soft}
              strokeLinejoin="round"
            />
            <path
              d="M0,0 L11,-6 L11,6 Z"
              fill={deep}
              {...soft}
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
        <Disc {...projection} center={center} normal={center} minDepth={0.3}>
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
    case "crown": {
      const crownLat = surfacePointAt(cx, top + 16, body).lat;
      const base: Vec3[] = Array.from({ length: 25 }, (_, index) =>
        fromSurface(-Math.PI + (2 * Math.PI * index) / 24, crownLat, 1.04),
      );
      const spikes = [-0.9, -0.3, 0.3, 0.9].map((lon) => [
        fromSurface(lon - 0.3, crownLat, 1.04),
        fromSurface(lon, crownLat + 0.42, 1.36),
        fromSurface(lon + 0.3, crownLat, 1.04),
      ]);
      return (
        <g>
          <Strand {...projection} deep={deep} points={base} width={5.5} />
          {spikes.map((spike, index) => (
            <Slab
              {...projection}
              deep={deep}
              outline={outline}
              key={index}
              points={spike}
            />
          ))}
        </g>
      );
    }
    case "propeller": {
      const capLat = surfacePointAt(cx, top + 14, body).lat;
      const capRadius = 1.08;
      const ring: Vec3[] = Array.from({ length: 41 }, (_, index) =>
        fromSurface(-Math.PI + (2 * Math.PI * index) / 40, capLat, capRadius),
      );
      const front = ring
        .map((point) => rotate(point, pose))
        .filter((point) => point[2] >= 0.02)
        .map((point) => toScreen(point, body));
      const crownScreen = toScreen(
        rotate([0, capRadius + 0.02, 0], pose),
        body,
      );
      const spin = ((pose.bob + 10) * 0.9 + pose.yaw * 6) % (Math.PI * 2);
      const blades: Vec3[] = [
        [Math.cos(spin) * 0.4, capRadius + 0.16, Math.sin(spin) * 0.4],
        [0, capRadius + 0.16, 0],
        [-Math.cos(spin) * 0.4, capRadius + 0.16, -Math.sin(spin) * 0.4],
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
          {dome ? (
            <path d={dome} fill={deep} {...edge} strokeLinejoin="round" />
          ) : null}
          <Strand
            {...projection}
            deep={deep}
            points={[
              [0, capRadius, 0],
              [0, capRadius + 0.16, 0],
            ]}
            width={2.4}
          />
          <Strand {...projection} deep={deep} points={blades} width={4} />
          {layer === "front" ? (
            <circle
              cx={crownScreen.x}
              cy={crownScreen.y - body.ry * 0.16}
              r={2.2}
              fill={deep}
              {...edge}
            />
          ) : null}
        </g>
      );
    }
    case "ears": {
      const ear = (side: number): Vec3[] => [
        fromSurface(side * 0.95, 0.55, 1.0),
        fromSurface(side * 0.75, 1.05, 1.42),
        fromSurface(side * 0.45, 0.85, 1.0),
      ];
      const inner = (side: number): Vec3[] => [
        fromSurface(side * 0.88, 0.62, 1.01),
        fromSurface(side * 0.74, 0.96, 1.3),
        fromSurface(side * 0.55, 0.83, 1.01),
      ];
      return (
        <g>
          {[-1, 1].map((side) => (
            <g key={side}>
              <Slab
                {...projection}
                deep={deep}
                outline={outline}
                points={ear(side)}
              />
              <Slab
                {...projection}
                deep={deep}
                outline={outline}
                points={inner(side)}
                fill="#fff"
              />
            </g>
          ))}
        </g>
      );
    }
    case "flower": {
      const center = fromSurface(-0.95, 0.8, 1.03);
      return (
        <Disc {...projection} center={center} normal={center} minDepth={0.5}>
          {[0, 72, 144, 216, 288].map((angle) => (
            <ellipse
              key={angle}
              cx={0}
              cy={-7.5}
              rx={4.6}
              ry={6.6}
              fill="#fff"
              {...edge}
              transform={`rotate(${angle})`}
            />
          ))}
          <circle r={4.2} fill={deep} {...edge} />
        </Disc>
      );
    }
    case "bowtie": {
      const chin = fromSurface(
        0,
        surfacePointAt(cx, bottom - 10, body).lat,
        1.02,
      );
      return (
        <Disc {...projection} center={chin} normal={chin} minDepth={0.35}>
          <g transform="scale(1.35)">
            <path
              d="M0,0 L-11,-6 L-11,6 Z"
              fill={deep}
              {...soft}
              strokeLinejoin="round"
            />
            <path
              d="M0,0 L11,-6 L11,6 Z"
              fill={deep}
              {...soft}
              strokeLinejoin="round"
            />
            <rect
              x={-2.6}
              y={-3}
              width={5.2}
              height={6}
              rx={1.5}
              fill={deep}
              {...edge}
            />
          </g>
        </Disc>
      );
    }
    case "headband": {
      const bandLat = surfacePointAt(cx, eyeY - 16, body).lat;
      const band: Vec3[] = Array.from({ length: 49 }, (_, index) =>
        fromSurface(-Math.PI + (2 * Math.PI * index) / 48, bandLat, 1.03),
      );
      const knot = fromSurface(-1.35, bandLat, 1.05);
      return (
        <g>
          <Strand {...projection} deep={deep} points={band} width={5} />
          <Strand
            {...projection}
            deep={deep}
            points={band}
            width={1.6}
            color="#fff"
          />
          <Disc {...projection} center={knot} normal={knot} minDepth={0.4}>
            <path
              d="M0,0 L-9,4 L-6,10 Z M0,0 L-2,10 L-7,14 Z"
              fill={deep}
              {...soft}
              strokeLinejoin="round"
            />
          </Disc>
        </g>
      );
    }
    default:
      return null;
  }
}
