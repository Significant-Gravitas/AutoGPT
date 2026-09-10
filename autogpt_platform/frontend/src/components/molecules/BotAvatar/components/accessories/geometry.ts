import { INK, type ShapeAnchors } from "../../helpers";
import { fromSurface, type Vec3 } from "../../parts";
import { surfacePointAt, type Ellipsoid, type Pose } from "../../projection";

export const HALF_PI = Math.PI / 2;
// Where a temple arm or an ear cup stops before it wraps past the silhouette.
export const TEMPLE_LON = Math.PI / 2 - 0.24;
export const GOLD = "#F5B942";
export const SHEEN = "#FFFFFF";

export function scaleVec([x, y, z]: Vec3, factor: number): Vec3 {
  return [x * factor, y * factor, z * factor];
}

export function ring(
  lat: number,
  radius: number,
  steps: number,
  fromLon = -Math.PI,
  toLon = Math.PI,
): Vec3[] {
  return Array.from({ length: steps + 1 }, (_, index) =>
    fromSurface(fromLon + ((toLon - fromLon) * index) / steps, lat, radius),
  );
}

// A horizontal disc — a hat brim, a halo — is edge-on when the head looks
// straight ahead, so it gets a drawn thickness that opens up as the head tips.
export function brimSquash(pose: Pose, base: number, gain: number) {
  return base + gain * Math.abs(Math.sin(pose.pitch));
}

export function eyeAnchors(anchors: ShapeAnchors, body: Ellipsoid) {
  return {
    eyeLat: surfacePointAt(anchors.cx, anchors.eyeY, body).lat,
    eyeLon: surfacePointAt(anchors.cx + anchors.eyeGap / 2, anchors.eyeY, body)
      .lon,
  };
}

export function latAtY(y: number, anchors: ShapeAnchors, body: Ellipsoid) {
  return surfacePointAt(anchors.cx, y, body).lat;
}

export function edgeProps(outline: boolean) {
  return outline ? { stroke: INK, strokeWidth: 2 } : {};
}

export function softProps(outline: boolean, deep: string) {
  return outline
    ? { stroke: INK, strokeWidth: 2 }
    : { stroke: deep, strokeWidth: 3 };
}

export const STAR_PATH =
  "M0,-11 L3.2,-3.4 L11.2,-3.2 L4.8,1.8 L7,9.6 L0,5 L-7,9.6 L-4.8,1.8 L-11.2,-3.2 L-3.2,-3.4 Z";

export const BOW_LEFT = "M0,0 L-12.5,-7.5 Q-15.5,0 -12.5,7.5 Z";
export const BOW_RIGHT = "M0,0 L12.5,-7.5 Q15.5,0 12.5,7.5 Z";

// Hats are drawn in screen units, so they scale with the head they sit on.
export function headUnit(body: { rx: number }) {
  return body.rx / 42;
}
