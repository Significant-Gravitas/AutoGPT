"use client";

import { GlassOrb } from "@/components/molecules/GlassOrb/GlassOrb";
import { GlassParams } from "@/components/molecules/GlassOrb/GlassSurface";
import { motion, type MotionValue, useTransform } from "framer-motion";
import { type AudioBarLevels } from "./useAudioBars";

interface Props {
  glassParams: GlassParams;
  audioBars: AudioBarLevels;
  isRecording: boolean;
}

export function OrbVisual({ glassParams, audioBars, isRecording }: Props) {
  return (
    <div data-testid="orb-current" className="h-full w-full">
      <GlassOrb params={glassParams} showRim={false}>
        <AudioBars levels={audioBars} isRecording={isRecording} />
      </GlassOrb>
    </div>
  );
}

const AUDIO_BARS = [
  { id: "low", height: 22, idleScale: 0.48 },
  { id: "low-mid", height: 34, idleScale: 0.58 },
  { id: "mid", height: 46, idleScale: 0.72 },
  { id: "high-mid", height: 34, idleScale: 0.58 },
  { id: "high", height: 22, idleScale: 0.48 },
];

function AudioBars({
  levels,
  isRecording,
}: {
  levels: AudioBarLevels;
  isRecording: boolean;
}) {
  return (
    <div
      data-testid="orb-audio-bars"
      className="flex h-12 items-center justify-center gap-1.5 drop-shadow-[0_2px_12px_rgba(90,40,180,0.5)]"
    >
      {AUDIO_BARS.map((bar, index) => (
        <AudioBar
          key={bar.id}
          level={levels[index]}
          height={bar.height}
          idleScale={bar.idleScale}
          isRecording={isRecording}
        />
      ))}
    </div>
  );
}

function AudioBar({
  level,
  height,
  idleScale,
  isRecording,
}: {
  level: MotionValue<number>;
  height: number;
  idleScale: number;
  isRecording: boolean;
}) {
  const reactiveScaleY = useTransform(level, [0, 1], [idleScale * 0.45, 1]);

  return (
    <motion.span
      data-testid="orb-audio-bar"
      className="block w-1.5 origin-center rounded-full bg-white/95"
      style={{ height, scaleY: isRecording ? reactiveScaleY : idleScale }}
    />
  );
}
