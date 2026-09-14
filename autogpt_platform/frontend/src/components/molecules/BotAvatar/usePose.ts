import { useAnimationFrame, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState, type RefObject } from "react";
import type { AvatarStatus } from "./helpers";
import { FRONT_POSE, type Pose } from "./projection";

const LOOK_REACH_PX = 320;
const LOOK_YAW = 0.55;
const LOOK_PITCH = 0.3;
const SMOOTHING = 0.12;

interface Args {
  status: AvatarStatus;
  animated: boolean;
  trackPointer: boolean;
  poseOffset?: Partial<Pose>;
  svgRef: RefObject<SVGSVGElement | null>;
}

const LOOK_WEIGHT: Record<AvatarStatus, number> = {
  idle: 1,
  thinking: 0.2,
  working: 0.25,
  waiting: 0.6,
  done: 0.9,
  failed: 0.3,
  sleeping: 0,
};

function statusPose(
  status: AvatarStatus,
  t: number,
  sinceChange: number,
): Pose {
  switch (status) {
    case "thinking":
      return {
        yaw: 0.35 + 0.12 * Math.sin(t * 0.9),
        pitch: -0.28 + 0.05 * Math.sin(t * 1.3),
        roll: -0.08 + 0.03 * Math.sin(t * 0.7),
        bob: 0.6 * Math.sin(t * 1.4),
      };
    case "failed": {
      const shake =
        sinceChange < 0.7
          ? Math.sin(sinceChange * 40) * (1 - sinceChange / 0.7)
          : 0;
      return {
        yaw: 0.35 * shake,
        pitch: 0.18 + 0.03 * Math.sin(t * 2),
        roll: 0.12 * shake,
        bob: -2 * (1 - Math.min(1, sinceChange / 0.7)),
      };
    }
    case "sleeping":
      return {
        yaw: 0.05 * Math.sin(t * 0.3),
        pitch: 0.3 + 0.04 * Math.sin(t * 0.7),
        roll: 0.16,
        bob: 1.2 * Math.sin(t * 0.7),
      };
    case "working":
      return {
        yaw: -0.45 + 0.06 * Math.sin(t * 7),
        pitch: 0.3 + 0.1 * Math.sin(t * 9),
        roll: 0.03 * Math.sin(t * 4.5),
        bob: 1.6 * Math.abs(Math.sin(t * 4.5)),
      };
    case "waiting":
      return {
        yaw: 0.75 * Math.sin(t * 1.3),
        pitch: -0.22 + 0.04 * Math.sin(t * 2.6),
        roll: 0.14 * Math.sin(t * 1.3 + 0.8),
        bob: 0,
      };
    case "done": {
      const nod =
        sinceChange < 0.9 ? Math.sin((sinceChange / 0.9) * Math.PI) : 0;
      return {
        yaw: 0.12 * Math.sin(t * 0.5),
        pitch: 0.35 * nod + 0.04 * Math.sin(t * 0.8),
        roll: 0,
        bob: 7 * nod + 0.8 * Math.sin(t * 1.6),
      };
    }
    default:
      return {
        yaw: 0.3 * Math.sin(t * 0.55),
        pitch: 0.08 * Math.sin(t * 0.85 + 1),
        roll: 0.02 * Math.sin(t * 0.4),
        bob: 1.2 * Math.sin(t * 1.7),
      };
  }
}

export function usePose({
  status,
  animated,
  trackPointer,
  poseOffset,
  svgRef,
}: Args) {
  const reducedMotion = useReducedMotion();
  const isLive = animated && !reducedMotion;
  const [pose, setPose] = useState<Pose>(FRONT_POSE);
  const look = useRef({ yaw: 0, pitch: 0 });
  const smoothed = useRef<Pose>(FRONT_POSE);
  const changedAt = useRef<number | null>(null);
  const clock = useRef(0);

  useEffect(() => {
    changedAt.current = clock.current;
  }, [status]);

  useEffect(() => {
    if (!isLive || !trackPointer) return;
    function handleMove(event: PointerEvent) {
      const box = svgRef.current?.getBoundingClientRect();
      if (!box) return;
      const dx = (event.clientX - (box.left + box.width / 2)) / LOOK_REACH_PX;
      const dy = (event.clientY - (box.top + box.height / 2)) / LOOK_REACH_PX;
      look.current = {
        yaw: Math.max(-1, Math.min(1, dx)) * LOOK_YAW,
        pitch: Math.max(-1, Math.min(1, dy)) * LOOK_PITCH,
      };
    }
    window.addEventListener("pointermove", handleMove);
    return () => window.removeEventListener("pointermove", handleMove);
  }, [isLive, trackPointer, svgRef]);

  useAnimationFrame((time) => {
    if (!isLive) return;
    const t = time / 1000;
    clock.current = t;
    const sinceChange = t - (changedAt.current ?? t);
    const base = statusPose(status, t, sinceChange);
    const weight = LOOK_WEIGHT[status];
    const target: Pose = {
      yaw: base.yaw + look.current.yaw * weight + (poseOffset?.yaw ?? 0),
      pitch:
        base.pitch + look.current.pitch * weight + (poseOffset?.pitch ?? 0),
      roll: base.roll + (poseOffset?.roll ?? 0),
      bob: base.bob,
    };
    const previous = smoothed.current;
    const next: Pose = {
      yaw: previous.yaw + (target.yaw - previous.yaw) * SMOOTHING,
      pitch: previous.pitch + (target.pitch - previous.pitch) * SMOOTHING,
      roll: previous.roll + (target.roll - previous.roll) * SMOOTHING,
      bob: target.bob,
    };
    smoothed.current = next;
    setPose(next);
  });

  const resolved = isLive
    ? pose
    : {
        ...FRONT_POSE,
        yaw: poseOffset?.yaw ?? 0,
        pitch: poseOffset?.pitch ?? 0,
        roll: poseOffset?.roll ?? 0,
      };

  return { isLive, pose: resolved };
}
