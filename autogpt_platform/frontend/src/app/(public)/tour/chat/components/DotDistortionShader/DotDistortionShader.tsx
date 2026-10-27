"use client";

import React, { useEffect, useRef, useCallback } from "react";
import { cn } from "@/lib/utils";

interface DotDistortionShaderProps {
  className?: string;
  dotGap?: number;
  dotSize?: number;
  dotColor?: string;
  glowColor?: string;
  backgroundColor?: string;
  mouseRadius?: number;
  distortionStrength?: number;
  breathingSpeed?: number;
  enableMouseInteraction?: boolean;
  opacity?: number;
  returnSpeed?: number;
  /** Render dots as a single static frame — no breathing, glow, or mouse-driven distortion. */
  isStatic?: boolean;
}

interface Dot {
  x: number;
  y: number;
  baseX: number;
  baseY: number;
  vx: number;
  vy: number;
  brightness: number;
  phase: number;
  breathingSpeed: number;
  glowIntensity: number;
  glowTarget: number;
  glowSpeed: number;
  nextGlowTime: number;
}

export const DotDistortionShader: React.FC<DotDistortionShaderProps> = ({
  className,
  dotGap = 16,
  dotSize = 1,
  dotColor = "var(--color-sky-700)",
  glowColor = "var(--color-sky-700)",
  backgroundColor,
  mouseRadius = 100,
  distortionStrength = 1,
  breathingSpeed = 1,
  enableMouseInteraction = true,
  opacity = 1,
  returnSpeed = 0.08,
  isStatic = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const dotsRef = useRef<Dot[]>([]);
  const mouseRef = useRef({ x: -1000, y: -1000 });
  const prevMouseRef = useRef({ x: -1000, y: -1000 });
  const mouseVelocityRef = useRef({ x: 0, y: 0 });
  const rafRef = useRef<number>(0);
  const timeRef = useRef<number>(0);

  const initDots = useCallback(
    (width: number, height: number) => {
      const dots: Dot[] = [];
      const cols = Math.ceil(width / dotGap) + 2;
      const rows = Math.ceil(height / dotGap) + 2;
      const offsetX = (width % dotGap) / 2;
      const offsetY = (height % dotGap) / 2;

      for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
          const x = i * dotGap + offsetX;
          const y = j * dotGap + offsetY;

          const noiseVal =
            Math.sin(i * 0.3 + j * 0.2) * 0.3 +
            Math.sin(i * 0.7 - j * 0.5) * 0.2 +
            Math.sin((i + j) * 0.4) * 0.2 +
            Math.random() * 0.3;

          const brightness = Math.max(0.1, Math.min(1, 0.3 + noiseVal));

          dots.push({
            x,
            y,
            baseX: x,
            baseY: y,
            vx: 0,
            vy: 0,
            brightness,
            phase: Math.random() * Math.PI * 2,
            breathingSpeed: 0.5 + Math.random() * 0.5,
            glowIntensity: 0,
            glowTarget: 0,
            glowSpeed: 0.002 + Math.random() * 0.003,
            nextGlowTime: Math.random() * 3,
          });
        }
      }
      return dots;
    },
    [dotGap],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resolveColor = (color: string) => {
      if (color.startsWith("var(")) {
        const varName = color.match(/var\(([^)]+)\)/)?.[1];
        if (varName) {
          return getComputedStyle(container).getPropertyValue(varName).trim();
        }
      }
      return color;
    };

    let stopped = false;
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    let resolvedDotColor = resolveColor(dotColor);
    let resolvedGlowColor = resolveColor(glowColor);

    const resize = () => {
      const { width, height } = container.getBoundingClientRect();
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${Math.floor(width)}px`;
      canvas.style.height = `${Math.floor(height)}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      dotsRef.current = initDots(width, height);
      resolvedDotColor = resolveColor(dotColor);
      resolvedGlowColor = resolveColor(glowColor);
      if (isStatic) requestAnimationFrame(drawStatic);
    };

    const drawStatic = () => {
      if (stopped) return;
      const { width, height } = container.getBoundingClientRect();
      ctx.clearRect(0, 0, width, height);
      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;
      ctx.fillStyle = resolvedDotColor;
      for (const dot of dotsRef.current) {
        ctx.globalAlpha = dot.brightness * opacity;
        ctx.beginPath();
        ctx.arc(dot.baseX, dot.baseY, dotSize, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    const ro = new ResizeObserver(resize);
    ro.observe(container);
    resize();
    if (isStatic) {
      drawStatic();
      return () => {
        stopped = true;
        ro.disconnect();
      };
    }

    let isFirstMouseEntry = true;

    const handleMouseMove = (e: MouseEvent) => {
      if (!enableMouseInteraction) return;
      const rect = container.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
        if (isFirstMouseEntry || mouseRef.current.x < 0) {
          mouseRef.current = { x, y };
          prevMouseRef.current = { x, y };
          mouseVelocityRef.current = { x: 0, y: 0 };
          isFirstMouseEntry = false;
        } else {
          prevMouseRef.current = { ...mouseRef.current };
          mouseRef.current = { x, y };

          mouseVelocityRef.current = {
            x: mouseRef.current.x - prevMouseRef.current.x,
            y: mouseRef.current.y - prevMouseRef.current.y,
          };
        }
      } else {
        mouseRef.current = { x: -1000, y: -1000 };
        mouseVelocityRef.current = { x: 0, y: 0 };
      }
    };

    window.addEventListener("mousemove", handleMouseMove);

    const handleVisibilityChange = () => {
      if (!document.hidden) {
        timeRef.current = 0;
        mouseVelocityRef.current = { x: 0, y: 0 };
        for (const dot of dotsRef.current) {
          dot.vx = 0;
          dot.vy = 0;
          dot.x = dot.baseX;
          dot.y = dot.baseY;
          dot.glowIntensity = 0;
          dot.glowTarget = 0;
          dot.nextGlowTime = Math.random() * 2;
        }
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);

    const draw = (timestamp: number) => {
      if (stopped) return;

      if (timeRef.current === 0) {
        timeRef.current = timestamp;
        rafRef.current = requestAnimationFrame(draw);
        return;
      }

      const dt = Math.min((timestamp - timeRef.current) / 16.67, 1.5);
      timeRef.current = timestamp;
      const { width, height } = container.getBoundingClientRect();

      ctx.clearRect(0, 0, width, height);
      ctx.globalAlpha = opacity;

      const time = timestamp * 0.001 * breathingSpeed;
      const timeSeconds = timestamp * 0.001;
      const mouseVelMagnitude = Math.sqrt(
        mouseVelocityRef.current.x ** 2 + mouseVelocityRef.current.y ** 2,
      );

      for (const dot of dotsRef.current) {
        if (timeSeconds >= dot.nextGlowTime) {
          if (dot.glowTarget === 0) {
            dot.glowTarget = 0.6 + Math.random() * 0.4;
            dot.glowSpeed = 0.001 + Math.random() * 0.002;
          } else {
            dot.glowTarget = 0;
            dot.glowSpeed = 0.0005 + Math.random() * 0.001;
            dot.nextGlowTime = timeSeconds + 1 + Math.random() * 3;
          }
        }

        const glowDiff = dot.glowTarget - dot.glowIntensity;
        dot.glowIntensity += glowDiff * dot.glowSpeed * dt * 60;

        if (dot.glowTarget > 0 && Math.abs(glowDiff) < 0.05) {
          dot.nextGlowTime = timeSeconds + 2 + Math.random() * 3;
        }

        const dx = mouseRef.current.x - dot.baseX;
        const dy = mouseRef.current.y - dot.baseY;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (
          distance < mouseRadius &&
          enableMouseInteraction &&
          mouseVelMagnitude > 0.5
        ) {
          const falloff = 1 - distance / mouseRadius;
          const strength = falloff * falloff * distortionStrength;

          dot.vx += mouseVelocityRef.current.x * strength * 0.3;
          dot.vy += mouseVelocityRef.current.y * strength * 0.3;
        }

        dot.x += dot.vx * dt;
        dot.y += dot.vy * dt;

        const springX = (dot.baseX - dot.x) * returnSpeed * dt;
        const springY = (dot.baseY - dot.y) * returnSpeed * dt;
        dot.x += springX;
        dot.y += springY;

        dot.vx *= 0.92;
        dot.vy *= 0.92;

        dot.vx += (dot.baseX - dot.x) * 0.02 * dt;
        dot.vy += (dot.baseY - dot.y) * 0.02 * dt;

        const breathingOffset =
          Math.sin(time * dot.breathingSpeed + dot.phase) * 0.15;
        const animatedBrightness = Math.max(
          0.05,
          Math.min(1, dot.brightness + breathingOffset),
        );

        const displacement = Math.sqrt(
          (dot.x - dot.baseX) ** 2 + (dot.y - dot.baseY) ** 2,
        );
        const brightnessBoost = Math.min(0.5, displacement * 0.05);

        const randomGlowBoost = dot.glowIntensity * 0.7;

        const finalBrightness = Math.min(
          1,
          animatedBrightness + brightnessBoost + randomGlowBoost,
        );

        const hasRandomGlow = dot.glowIntensity > 0.1;
        if (finalBrightness > 0.4 || hasRandomGlow) {
          const baseGlowIntensity = (finalBrightness - 0.4) / 0.6;
          const combinedGlowIntensity = Math.max(
            baseGlowIntensity,
            dot.glowIntensity,
          );
          ctx.shadowColor = resolvedGlowColor;
          ctx.shadowBlur = 10 + 20 * combinedGlowIntensity;
        } else {
          ctx.shadowColor = "transparent";
          ctx.shadowBlur = 0;
        }

        ctx.globalAlpha = finalBrightness * opacity;
        ctx.fillStyle = resolvedDotColor;
        ctx.beginPath();
        ctx.arc(dot.x, dot.y, dotSize, 0, Math.PI * 2);
        ctx.fill();
      }

      mouseVelocityRef.current.x *= 0.9;
      mouseVelocityRef.current.y *= 0.9;

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);

    return () => {
      stopped = true;
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      ro.disconnect();
    };
  }, [
    dotGap,
    dotSize,
    dotColor,
    glowColor,
    mouseRadius,
    distortionStrength,
    breathingSpeed,
    enableMouseInteraction,
    opacity,
    returnSpeed,
    isStatic,
    initDots,
  ]);

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative overflow-hidden bg-white dark:bg-black",
        className,
      )}
      style={{
        background: backgroundColor,
      }}
    >
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full"
        style={{ display: "block" }}
      />
    </div>
  );
};
