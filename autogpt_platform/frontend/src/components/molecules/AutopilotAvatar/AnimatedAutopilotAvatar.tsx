"use client";

import { cn } from "@/lib/utils";
import type { AvatarStatus } from "@/components/molecules/NotionAvatar/expressions";
import {
  cssHeadTransform,
  type Pose,
} from "@/components/molecules/NotionAvatar/pose";
import { usePose } from "@/components/molecules/NotionAvatar/usePose";
import Image from "next/image";
import { useRef } from "react";
import {
  AUTOPILOT_AVATAR_BG_CLASS,
  AUTOPILOT_AVATAR_URL,
  AUTOPILOT_NAME,
} from "./helpers";

interface Props {
  status?: AvatarStatus;
  size?: number;
  animated?: boolean;
  trackPointer?: boolean;
  poseOffset?: Partial<Pose>;
  className?: string;
}

/** Otto at scene size. He keeps his own drawing rather than a generated face,
 *  so he sways and follows the pointer but does not blink or change
 *  expression — there are no separate eye or mouth layers to swap. */
export function AnimatedAutopilotAvatar({
  status = "idle",
  size = 120,
  animated = true,
  trackPointer = false,
  poseOffset,
  className,
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const { pose } = usePose({
    status,
    animated,
    trackPointer,
    poseOffset,
    svgRef: hostRef,
  });

  return (
    <div
      ref={hostRef}
      style={{ width: size, height: size }}
      data-testid="autopilot-avatar"
      data-status={status}
      className={cn(
        "relative shrink-0 overflow-hidden rounded-full border border-stone-500",
        AUTOPILOT_AVATAR_BG_CLASS,
        className,
      )}
    >
      <Image
        src={AUTOPILOT_AVATAR_URL}
        alt={AUTOPILOT_NAME}
        width={size}
        height={size}
        priority
        style={{ transform: cssHeadTransform(pose) }}
      />
    </div>
  );
}
