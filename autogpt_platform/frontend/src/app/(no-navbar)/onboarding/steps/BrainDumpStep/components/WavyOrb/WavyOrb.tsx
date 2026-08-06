"use client";

import { type WavyOrbSettings } from "./helpers";
import { useWavyOrb } from "./useWavyOrb";

interface Props {
  audioStream: MediaStream | null;
  settings: WavyOrbSettings;
}

export function WavyOrb({ audioStream, settings }: Props) {
  const { canvasRef, isSupported } = useWavyOrb(audioStream, settings);

  if (!isSupported) {
    return (
      <div
        data-testid="orb-wavy"
        className="relative h-full w-full overflow-hidden"
        aria-hidden
      >
        <div className="absolute left-0 top-1/2 h-1 w-full -translate-y-1/2 bg-gradient-to-r from-transparent via-purple-500 to-transparent blur-sm" />
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      data-testid="orb-wavy"
      className="block h-full w-full bg-transparent"
      aria-hidden
    />
  );
}
