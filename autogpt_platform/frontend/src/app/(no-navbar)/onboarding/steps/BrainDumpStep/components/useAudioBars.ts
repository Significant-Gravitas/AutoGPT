"use client";

import { useEffect } from "react";
import { type MotionValue, useMotionValue } from "framer-motion";

export type AudioBarLevels = MotionValue<number>[];

const VOICE_BANDS = [
  [80, 250],
  [250, 500],
  [500, 900],
  [900, 1600],
  [1600, 3000],
] as const;

export function useAudioBars(audioStream: MediaStream | null) {
  const first = useMotionValue(0);
  const second = useMotionValue(0);
  const third = useMotionValue(0);
  const fourth = useMotionValue(0);
  const fifth = useMotionValue(0);

  useEffect(() => {
    const levels = [first, second, third, fourth, fifth];

    if (!audioStream) {
      levels.forEach((level) => level.set(0));
      return;
    }

    const audioContext = new AudioContext();
    if (audioContext.state === "suspended") {
      void audioContext.resume();
    }
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.45;
    const source = audioContext.createMediaStreamSource(audioStream);
    source.connect(analyser);
    const samples = new Uint8Array(analyser.frequencyBinCount);
    const binWidth = audioContext.sampleRate / analyser.fftSize;
    let animationFrame = 0;
    const currentLevels = [0, 0, 0, 0, 0];
    let lastFrame = performance.now();

    function update(now: number) {
      const delta = Math.min((now - lastFrame) / 1000, 0.1);
      lastFrame = now;
      analyser.getByteFrequencyData(samples);

      VOICE_BANDS.forEach(([minimum, maximum], index) => {
        const start = Math.max(1, Math.ceil(minimum / binWidth));
        const end = Math.min(samples.length, Math.ceil(maximum / binWidth));
        let energy = 0;

        for (let sampleIndex = start; sampleIndex < end; sampleIndex += 1) {
          const normalized = samples[sampleIndex] / 255;
          energy += normalized * normalized;
        }

        const rms = Math.sqrt(energy / Math.max(1, end - start));
        const target = Math.min(1, Math.max(0, (rms - 0.025) * 2.2));
        const rate = target > currentLevels[index] ? 18 : 7;
        currentLevels[index] +=
          (target - currentLevels[index]) * Math.min(1, delta * rate);
        levels[index].set(currentLevels[index]);
      });

      animationFrame = requestAnimationFrame(update);
    }

    animationFrame = requestAnimationFrame(update);

    return () => {
      cancelAnimationFrame(animationFrame);
      source.disconnect();
      levels.forEach((level) => level.set(0));
      void audioContext.close().catch(() => undefined);
    };
  }, [audioStream, fifth, first, fourth, second, third]);

  return [first, second, third, fourth, fifth];
}
