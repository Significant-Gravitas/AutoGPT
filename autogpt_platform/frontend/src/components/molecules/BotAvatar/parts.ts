import type { Ellipsoid, Pose, Projected } from "./projection";

export type Vec3 = [number, number, number];
export type Layer = "front" | "back";

export function fromSurface(lon: number, lat: number, radius = 1): Vec3 {
  const cosLat = Math.cos(lat);
  return [
    radius * cosLat * Math.sin(lon),
    radius * Math.sin(lat),
    radius * cosLat * Math.cos(lon),
  ];
}

export function rotate([x0, y0, z0]: Vec3, pose: Pose): Vec3 {
  const yawCos = Math.cos(pose.yaw);
  const yawSin = Math.sin(pose.yaw);
  const x = x0 * yawCos + z0 * yawSin;
  const z1 = -x0 * yawSin + z0 * yawCos;
  const pitchCos = Math.cos(pose.pitch);
  const pitchSin = Math.sin(pose.pitch);
  return [x, y0 * pitchCos - z1 * pitchSin, y0 * pitchSin + z1 * pitchCos];
}

export function toScreen([x, y]: Vec3, body: Ellipsoid) {
  return { x: body.cx + body.rx * x, y: body.cy - body.ry * y };
}

export function surfaceDepth([x, y]: Vec3) {
  return Math.sqrt(Math.max(0, 1 - x * x - y * y));
}

export function isInFront(point: Vec3, tolerance = 0.015) {
  return point[2] >= surfaceDepth(point) - tolerance;
}

export function layerOf(point: Vec3): Layer {
  return isInFront(point) ? "front" : "back";
}

export function discProjection(
  center: Vec3,
  normal: Vec3,
  pose: Pose,
  body: Ellipsoid,
): Projected {
  const c = rotate(center, pose);
  const n = rotate(normal, pose);
  const screen = toScreen(c, body);
  return {
    x: screen.x,
    y: screen.y,
    depth: Math.abs(n[2]),
    angle: (Math.atan2(-n[1], n[0]) * 180) / Math.PI,
  };
}

export function discTransform(projected: Projected, minDepth = 0.12) {
  const depth = Math.max(minDepth, projected.depth);
  return `translate(${projected.x} ${projected.y}) rotate(${projected.angle}) scale(${depth} 1) rotate(${-projected.angle})`;
}

export function samplePolyline(points: Vec3[], perSegment = 6): Vec3[] {
  const out: Vec3[] = [];
  for (let index = 0; index < points.length - 1; index += 1) {
    const [ax, ay, az] = points[index];
    const [bx, by, bz] = points[index + 1];
    for (let step = 0; step < perSegment; step += 1) {
      const t = step / perSegment;
      out.push([ax + (bx - ax) * t, ay + (by - ay) * t, az + (bz - az) * t]);
    }
  }
  out.push(points[points.length - 1]);
  return out;
}

export function splitPolyline(points: Vec3[], pose: Pose, body: Ellipsoid) {
  const rotated = points.map((point) => rotate(point, pose));
  const paths: Record<Layer, string[]> = { front: [], back: [] };
  let previous: Layer | null = null;
  rotated.forEach((point) => {
    const layer = layerOf(point);
    const { x, y } = toScreen(point, body);
    const command = layer === previous ? "L" : "M";
    if (layer !== previous && previous !== null) {
      paths[previous].push(`L${x.toFixed(2)},${y.toFixed(2)}`);
    }
    paths[layer].push(`${command}${x.toFixed(2)},${y.toFixed(2)}`);
    previous = layer;
  });
  return { front: paths.front.join(" "), back: paths.back.join(" ") };
}

export function polygonPath(points: Vec3[], pose: Pose, body: Ellipsoid) {
  const screen = points.map((point) => toScreen(rotate(point, pose), body));
  return `M${screen.map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(" L")} Z`;
}

export function centroidLayer(points: Vec3[], pose: Pose): Layer {
  const rotated = points.map((point) => rotate(point, pose));
  const center = rotated.reduce<Vec3>(
    (sum, [x, y, z]) => [
      sum[0] + x / rotated.length,
      sum[1] + y / rotated.length,
      sum[2] + z / rotated.length,
    ],
    [0, 0, 0],
  );
  return layerOf(center);
}
