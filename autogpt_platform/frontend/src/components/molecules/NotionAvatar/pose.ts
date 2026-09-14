import { VIEWBOX } from "./metadata.generated";

export interface Pose {
  yaw: number;
  pitch: number;
  roll: number;
  bob: number;
}

export const FRONT_POSE: Pose = { yaw: 0, pitch: 0, roll: 0, bob: 0 };

// The artwork is flat, so a head turn is a shift rather than a rotation about
// a sphere. These are viewBox units on the 1080 canvas, tuned so a full-weight
// look reads as a glance without sliding the face off its disc.
const YAW_SHIFT = 70;
const PITCH_SHIFT = 55;
const BOB_SHIFT = 9;
// The features ride further than the head, which is what sells the turn.
const PARALLAX = 0.38;

function transform(pose: Pose, factor: number, centre: number): string {
  const x = pose.yaw * YAW_SHIFT * factor;
  const y = pose.pitch * PITCH_SHIFT * factor - pose.bob * BOB_SHIFT * factor;
  const degrees = (pose.roll * 180) / Math.PI;
  return `translate(${x.toFixed(2)} ${y.toFixed(2)}) rotate(${degrees.toFixed(2)} ${centre} ${centre})`;
}

export function headTransform(pose: Pose, centre: number): string {
  return transform(pose, 1, centre);
}

/** Applied on top of the head's own transform, so the eyes lead the turn. */
export function featureTransform(pose: Pose): string {
  const x = pose.yaw * YAW_SHIFT * PARALLAX;
  const y = pose.pitch * PITCH_SHIFT * PARALLAX;
  return `translate(${x.toFixed(2)} ${y.toFixed(2)})`;
}

/** The same motion as a CSS transform, for artwork that is not built out of
 *  swappable layers — Otto, who keeps his own drawing. Percentages so it holds
 *  at any rendered size. */
export function cssHeadTransform(pose: Pose): string {
  const x = ((pose.yaw * YAW_SHIFT) / VIEWBOX) * 100;
  const y = ((pose.pitch * PITCH_SHIFT - pose.bob * BOB_SHIFT) / VIEWBOX) * 100;
  const degrees = (pose.roll * 180) / Math.PI;
  return `translate(${x.toFixed(2)}%, ${y.toFixed(2)}%) rotate(${degrees.toFixed(2)}deg)`;
}
