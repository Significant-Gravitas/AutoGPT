"use client";

import { useState } from "react";
import { type MotionValue, useMotionValueEvent } from "framer-motion";
import { Orb, type OrbState } from "orb-ui";

export const ORB_UI_SIZE = 280;

interface Props {
  audioLevel: MotionValue<number>;
  state: OrbState;
}

export function OrbUiOrb({ audioLevel, state }: Props) {
  const [volume, setVolume] = useState(audioLevel.get());
  const isIdlePreview = state === "idle";

  useMotionValueEvent(audioLevel, "change", (latest) => {
    setVolume(Math.round(latest * 100) / 100);
  });

  return (
    <Orb
      data-testid="orb-ui"
      theme="cloud"
      state={isIdlePreview ? "listening" : state}
      volume={isIdlePreview ? 0.08 : volume}
      size={ORB_UI_SIZE}
      interactive={false}
    />
  );
}
