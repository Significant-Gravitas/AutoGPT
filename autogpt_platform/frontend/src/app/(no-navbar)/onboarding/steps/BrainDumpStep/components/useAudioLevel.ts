"use client";

import { useEffect } from "react";
import { useMotionValue } from "framer-motion";

export function useAudioLevel(audioStream: MediaStream | null) {
  const level = useMotionValue(0);

  useEffect(() => {
    if (!audioStream) {
      level.set(0);
      return;
    }

    const audioContext = new AudioContext();
    if (audioContext.state === "suspended") {
      void audioContext.resume();
    }
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.55;
    audioContext.createMediaStreamSource(audioStream).connect(analyser);
    const samples = new Uint8Array(analyser.fftSize);
    let animationFrame = 0;
    let currentLevel = 0;
    let lastFrame = performance.now();

    function update(now: number) {
      const delta = Math.min((now - lastFrame) / 1000, 0.1);
      lastFrame = now;
      analyser.getByteTimeDomainData(samples);

      let sum = 0;
      for (const sample of samples) {
        const normalized = (sample - 128) / 128;
        sum += normalized * normalized;
      }

      const rms = Math.sqrt(sum / samples.length);
      const target = Math.min(1, Math.max(0, (rms - 0.01) * 12));
      const rate = target > currentLevel ? 16 : 4.5;
      currentLevel += (target - currentLevel) * Math.min(1, delta * rate);
      level.set(currentLevel);
      animationFrame = requestAnimationFrame(update);
    }

    animationFrame = requestAnimationFrame(update);

    return () => {
      cancelAnimationFrame(animationFrame);
      level.set(0);
      void audioContext.close();
    };
  }, [audioStream, level]);

  return level;
}
