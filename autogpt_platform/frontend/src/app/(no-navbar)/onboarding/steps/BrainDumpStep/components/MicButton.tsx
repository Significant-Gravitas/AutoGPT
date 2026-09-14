"use client";

import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import {
  AUTOPILOT_AURA_COLORS,
  AUTOPILOT_DOT_COLOR,
} from "@/components/molecules/AutopilotAvatar/helpers";
import {
  animate,
  type AnimationPlaybackControls,
  motion,
  useMotionValue,
  useReducedMotion,
  useTransform,
} from "framer-motion";
import { useEffect, useState } from "react";
import { useAudioBars } from "./useAudioBars";
import { useSimulatedVoice } from "./useSimulatedVoice";
import { VoiceAura } from "./VoiceAura";
import { DOT, VoiceDots } from "./VoiceDots";

export type OrbScreen = "rest" | "recording" | "processing" | "failed";

const AVATAR_SIZE = 160;
// The body collapses into, and grows back out of, the middle of its disc.
const CENTRE = { x: AVATAR_SIZE / 2, y: AVATAR_SIZE / 2 };
const DOT_COLOR = AUTOPILOT_DOT_COLOR;
// Quick and near-critically damped: fast in, a whisper of settle, no wobble.
const COLLAPSE = {
  type: "spring",
  stiffness: 520,
  damping: 32,
  mass: 0.7,
} as const;
const SPLIT = {
  type: "spring",
  stiffness: 560,
  damping: 30,
  mass: 0.7,
} as const;
// A full tumble on the avatar's pitch axis, overlapping the fold so the
// body is mid-flip as it becomes the dot.
const REDUCED = { duration: 0.15 } as const;
// The next stage starts this soon after the previous one begins, so the
// stages overlap instead of waiting for each bounce to die down.
const OVERLAP_MS = 120;

interface Props {
  screen: OrbScreen;
  audioStream: MediaStream | null;
  // Fakes a voice so the dots can be seen moving without a microphone.
  simulateVoice?: boolean;
}

// Otto is the whole visual. Opening the mic folds the body into a
// single dot at its centre, with the comet wave orbiting only while it
// folds; the dot then splits into a row of four that move with the voice.
// Closing the mic runs it backwards: the dots merge, then the body grows
// back with the wave around it.
export function MicButton({
  screen,
  audioStream,
  simulateVoice = false,
}: Props) {
  const isRecording = screen === "recording";
  const reduceMotion = useReducedMotion() === true;
  const micLevels = useAudioBars(isRecording ? audioStream : null);
  const fakeLevels = useSimulatedVoice(isRecording && simulateVoice);
  const levels = simulateVoice ? fakeLevels : micLevels;

  const collapse = useMotionValue(0);
  const split = useMotionValue(0);
  const [isFolding, setIsFolding] = useState(false);

  useEffect(() => {
    const target = isRecording ? 1 : 0;
    if (collapse.get() === target && split.get() === target) {
      setIsFolding(false);
      return;
    }
    let pauseTimer: ReturnType<typeof setTimeout> | undefined;
    const running: AnimationPlaybackControls[] = [];
    let cancelled = false;
    const step = (value: typeof collapse, to: number, physics: object) => {
      const controls = animate(value, to, reduceMotion ? REDUCED : physics);
      running.push(controls);
      return controls;
    };
    const pause = () =>
      new Promise<void>((resolve) => {
        pauseTimer = setTimeout(resolve, OVERLAP_MS);
      });
    async function fold() {
      setIsFolding(true);
      step(collapse, 1, COLLAPSE).then(() => {
        if (!cancelled) setIsFolding(false);
      });
      await pause();
      if (cancelled) return;
      step(split, 1, SPLIT);
    }
    async function unfold() {
      step(split, 0, SPLIT);
      await pause();
      if (cancelled) return;
      setIsFolding(true);
      step(collapse, 0, COLLAPSE).then(() => {
        if (!cancelled) setIsFolding(false);
      });
    }
    void (isRecording ? fold() : unfold());
    return () => {
      cancelled = true;
      clearTimeout(pauseTimer);
      running.forEach((controls) => controls.stop());
    };
  }, [isRecording, reduceMotion, collapse, split]);

  const bodyScale = useTransform(collapse, [0, 1], [1, DOT / AVATAR_SIZE]);
  const bodyOpacity = useTransform(collapse, [0, 0.55, 1], [1, 1, 0]);

  return (
    <div data-testid="autopilot-avatar" data-screen={screen}>
      <VoiceAura
        colors={AUTOPILOT_AURA_COLORS}
        size={AVATAR_SIZE}
        levels={levels}
        isActive={isFolding && !reduceMotion}
      >
        <motion.div
          style={{
            scale: bodyScale,
            opacity: bodyOpacity,
            transformOrigin: `${CENTRE.x}px ${CENTRE.y}px`,
            willChange: "transform, opacity",
          }}
        >
          <AutopilotAvatar size={AVATAR_SIZE} />
        </motion.div>
        <VoiceDots
          levels={levels}
          collapse={collapse}
          split={split}
          centre={CENTRE}
          color={DOT_COLOR}
          reduceMotion={reduceMotion}
        />
      </VoiceAura>
    </div>
  );
}
