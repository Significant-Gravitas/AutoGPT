import type { ShapeAnchors } from "./helpers";

export interface Pose {
  yaw: number;
  pitch: number;
  roll: number;
  bob: number;
}

export interface SurfacePoint {
  lon: number;
  lat: number;
}

export interface Projected {
  x: number;
  y: number;
  depth: number;
  angle: number;
}

export interface Ellipsoid {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
}

export const FRONT_POSE: Pose = { yaw: 0, pitch: 0, roll: 0, bob: 0 };

export function ellipsoidFor(anchors: ShapeAnchors): Ellipsoid {
  return {
    cx: anchors.cx,
    cy: (anchors.top + anchors.bottom) / 2,
    rx: anchors.width / 2,
    ry: (anchors.bottom - anchors.top) / 2,
  };
}

export function surfacePointAt(
  x: number,
  y: number,
  body: Ellipsoid,
): SurfacePoint {
  const lat = Math.asin(clamp((body.cy - y) / body.ry, -1, 1));
  const lon = Math.asin(
    clamp((x - body.cx) / (body.rx * Math.cos(lat)), -1, 1),
  );
  return { lon, lat };
}

export function project(
  point: SurfacePoint,
  pose: Pose,
  body: Ellipsoid,
): Projected {
  const cosLat = Math.cos(point.lat);
  let x = cosLat * Math.sin(point.lon);
  let y = Math.sin(point.lat);
  let z = cosLat * Math.cos(point.lon);

  const yawCos = Math.cos(pose.yaw);
  const yawSin = Math.sin(pose.yaw);
  [x, z] = [x * yawCos + z * yawSin, -x * yawSin + z * yawCos];

  const pitchCos = Math.cos(pose.pitch);
  const pitchSin = Math.sin(pose.pitch);
  [y, z] = [y * pitchCos - z * pitchSin, y * pitchSin + z * pitchCos];

  return {
    x: body.cx + body.rx * x,
    y: body.cy - body.ry * y,
    depth: z,
    angle: (Math.atan2(-y, x) * 180) / Math.PI,
  };
}

export function foreshortenTransform(projected: Projected, minDepth = 0) {
  const depth = Math.max(minDepth, projected.depth);
  return `translate(${projected.x} ${projected.y}) rotate(${projected.angle}) scale(${depth} 1) rotate(${-projected.angle})`;
}

export function isVisible(projected: Projected, threshold = 0.02) {
  return projected.depth > threshold;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

export function arcPath(
  points: SurfacePoint[],
  pose: Pose,
  body: Ellipsoid,
  minDepth = -0.05,
) {
  const segments: string[] = [];
  let open = false;
  for (const point of points) {
    const projected = project(point, pose, body);
    if (projected.depth < minDepth) {
      open = false;
      continue;
    }
    segments.push(
      `${open ? "L" : "M"}${projected.x.toFixed(2)},${projected.y.toFixed(2)}`,
    );
    open = true;
  }
  return segments.join(" ");
}

export function latitudeRing(
  lat: number,
  steps: number,
  fromLon = -Math.PI,
  toLon = Math.PI,
): SurfacePoint[] {
  return Array.from({ length: steps + 1 }, (_, index) => ({
    lon: fromLon + ((toLon - fromLon) * index) / steps,
    lat,
  }));
}

export function pointFromVector(x: number, y: number, z: number): SurfacePoint {
  const length = Math.hypot(x, y, z) || 1;
  return {
    lat: Math.asin(y / length),
    lon: Math.atan2(x / length, z / length),
  };
}

export function coronalArc(
  fromLat: number,
  steps: number,
  forward: number,
): SurfacePoint[] {
  return Array.from({ length: steps + 1 }, (_, index) => {
    const theta = Math.PI - fromLat - ((Math.PI - 2 * fromLat) * index) / steps;
    return pointFromVector(Math.cos(theta), Math.sin(theta), forward);
  });
}
