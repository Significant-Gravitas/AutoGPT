"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { GlassOrb } from "@/components/molecules/GlassOrb/GlassOrb";
import { GlassParams } from "@/components/molecules/GlassOrb/GlassSurface";
import { AudioWave01Icon } from "@hugeicons/core-free-icons";
import { motion, type MotionValue, useTransform } from "framer-motion";
import { ORB_UI_SIZE, OrbUiOrb } from "./OrbUiOrb";
import { OrbVariant } from "./OrbSelector";
import { WavyOrb, type WavyOrbSettings } from "./WavyOrb";

interface Props {
  variant: OrbVariant;
  glassParams: GlassParams;
  audioStream: MediaStream | null;
  audioLevel: MotionValue<number>;
  wavySettings: WavyOrbSettings;
  isRecording: boolean;
  isLoading: boolean;
}

export function OrbVisual({
  variant,
  glassParams,
  audioStream,
  audioLevel,
  wavySettings,
  isRecording,
  isLoading,
}: Props) {
  if (variant === "glass") {
    return (
      <div data-testid="orb-current" className="h-full w-full">
        <GlassOrb params={glassParams} audioLevel={audioLevel} showRim={false}>
          <ReactiveWaveIcon audioLevel={audioLevel} />
        </GlassOrb>
      </div>
    );
  }

  if (variant === "orb-ui") {
    return (
      <div className="relative h-full w-full" aria-hidden>
        <div
          data-testid="orb-ui-frame"
          className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
          style={{ width: ORB_UI_SIZE, height: ORB_UI_SIZE }}
        >
          <OrbUiOrb
            audioLevel={audioLevel}
            state={isLoading ? "thinking" : isRecording ? "listening" : "idle"}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="relative h-full w-full" aria-hidden>
      <div className="absolute left-1/2 top-1/2 h-[240px] w-[620px] -translate-x-1/2 -translate-y-1/2">
        <WavyOrb audioStream={audioStream} settings={wavySettings} />
      </div>
    </div>
  );
}

function ReactiveWaveIcon({ audioLevel }: { audioLevel: MotionValue<number> }) {
  const scale = useTransform(audioLevel, [0, 1], [1, 1.2]);

  return (
    <motion.div
      data-testid="orb-voice-wave"
      style={{ scale }}
      className="flex items-center justify-center text-white/95 drop-shadow-[0_2px_12px_rgba(90,40,180,0.5)]"
    >
      <Icon icon={AudioWave01Icon} size={56} strokeWidth={1.5} />
    </motion.div>
  );
}
