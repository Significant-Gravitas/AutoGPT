"use client";

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";
import { createWavyOrbRenderer, type WavyOrbSettings } from "./helpers";

export function useWavyOrb(
  audioStream: MediaStream | null,
  settings: WavyOrbSettings,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const settingsRef = useRef(settings);
  const prefersReducedMotion = useReducedMotion();
  const [isSupported, setIsSupported] = useState(true);

  useEffect(() => {
    settingsRef.current = settings;
  }, [settings]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const renderer = createWavyOrbRenderer({
      canvas,
      audioStream,
      getSettings: () => settingsRef.current,
      prefersReducedMotion: Boolean(prefersReducedMotion),
    });
    if (!renderer) {
      setIsSupported(false);
      return;
    }

    let animationFrame = 0;
    let isVisible = true;
    let isIntersecting = true;

    function render(now: number) {
      if (!isVisible || !isIntersecting) return;
      renderer.draw(now);
      animationFrame = requestAnimationFrame(render);
    }

    function syncAnimation() {
      isVisible = document.visibilityState !== "hidden";
      cancelAnimationFrame(animationFrame);
      if (!isVisible || !isIntersecting) return;
      if (prefersReducedMotion) {
        renderer.draw(performance.now());
        return;
      }
      animationFrame = requestAnimationFrame(render);
    }

    const resizeObserver = new ResizeObserver(() =>
      renderer.draw(performance.now()),
    );
    const intersectionObserver = new IntersectionObserver(([entry]) => {
      isIntersecting = entry.isIntersecting;
      syncAnimation();
    });
    resizeObserver.observe(canvas);
    intersectionObserver.observe(canvas);
    document.addEventListener("visibilitychange", syncAnimation);
    syncAnimation();

    return () => {
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      document.removeEventListener("visibilitychange", syncAnimation);
      renderer.dispose();
    };
  }, [audioStream, prefersReducedMotion]);

  return { canvasRef, isSupported };
}
