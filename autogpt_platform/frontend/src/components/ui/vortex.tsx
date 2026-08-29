"use client";
import { cn } from "@/lib/utils";
import { ReactNode, useRef } from "react";
import { createNoise3D } from "simplex-noise";
import { motion } from "motion/react";
import { useMountEffect } from "@/hooks/useMountEffect";

interface VortexProps {
  children?: ReactNode;
  className?: string;
  containerClassName?: string;
  particleCount?: number;
  rangeY?: number;
  baseHue?: number;
  baseSpeed?: number;
  rangeSpeed?: number;
  baseRadius?: number;
  rangeRadius?: number;
  backgroundColor?: string;
}

export function Vortex(props: VortexProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const animationFrameId = useRef<number>();
  const particleCount = props.particleCount || 700;
  const particlePropCount = 9;
  const particlePropsLength = particleCount * particlePropCount;
  const rangeY = props.rangeY || 100;
  const baseTTL = 50;
  const rangeTTL = 150;
  const baseSpeed = props.baseSpeed || 0.0;
  const rangeSpeed = props.rangeSpeed || 1.5;
  const baseRadius = props.baseRadius || 1;
  const rangeRadius = props.rangeRadius || 2;
  const baseHue = props.baseHue || 220;
  const rangeHue = 100;
  const noiseSteps = 3;
  const xOff = 0.00125;
  const yOff = 0.00125;
  const zOff = 0.0005;
  const backgroundColor = props.backgroundColor || "#000000";
  let tick = 0;
  let lastTime = 0;
  let dtFrames = 1;
  let isRunning = false;
  const noise3D = createNoise3D();
  let particleProps = new Float32Array(particlePropsLength);
  const center: [number, number] = [0, 0];

  const TAU: number = 2 * Math.PI;

  function rand(n: number): number {
    return n * Math.random();
  }

  function randRange(n: number): number {
    return n - rand(2 * n);
  }

  function fadeInOut(t: number, m: number): number {
    const hm = 0.5 * m;
    return Math.abs(((t + hm) % m) - hm) / hm;
  }

  function lerp(n1: number, n2: number, speed: number): number {
    return (1 - speed) * n1 + speed * n2;
  }

  function initParticles() {
    tick = 0;
    particleProps = new Float32Array(particlePropsLength);

    for (let i = 0; i < particlePropsLength; i += particlePropCount) {
      initParticle(i);
    }
  }

  function initParticle(i: number) {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const x = rand(canvas.width);
    const y = center[1] + randRange(rangeY);
    const vx = 0;
    const vy = 0;
    const life = 0;
    const ttl = baseTTL + rand(rangeTTL);
    const speed = baseSpeed + rand(rangeSpeed);
    const radius = baseRadius + rand(rangeRadius);
    const hue = baseHue + rand(rangeHue);

    particleProps.set([x, y, vx, vy, life, ttl, speed, radius, hue], i);
  }

  function startLoop() {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || isRunning) return;

    isRunning = true;
    lastTime = performance.now();
    animationFrameId.current = window.requestAnimationFrame(() =>
      loop(canvas, ctx),
    );
  }

  function stopLoop() {
    isRunning = false;
    if (animationFrameId.current) {
      window.cancelAnimationFrame(animationFrameId.current);
    }
  }

  function loop(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D) {
    const now = performance.now();
    const dt = Math.min((now - lastTime) / 1000, 1 / 30);
    lastTime = now;
    dtFrames = dt * 60;
    tick += dtFrames;

    renderFrame(canvas, ctx);

    if (isRunning) {
      animationFrameId.current = window.requestAnimationFrame(() =>
        loop(canvas, ctx),
      );
    }
  }

  function renderFrame(
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D,
  ) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.lineCap = "round";
    ctx.globalCompositeOperation = "lighter";
    stepParticles(ctx);
    ctx.globalCompositeOperation = "source-over";
  }

  function drawStaticFrame(
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D,
  ) {
    dtFrames = 1;
    for (let i = 0; i < 90; i++) {
      tick += 1;
      stepParticles(null);
    }
    renderFrame(canvas, ctx);
  }

  function stepParticles(ctx: CanvasRenderingContext2D | null) {
    for (let i = 0; i < particlePropsLength; i += particlePropCount) {
      updateParticle(i, ctx);
    }
  }

  function updateParticle(i: number, ctx: CanvasRenderingContext2D | null) {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const i2 = 1 + i,
      i3 = 2 + i,
      i4 = 3 + i,
      i5 = 4 + i,
      i6 = 5 + i,
      i7 = 6 + i,
      i8 = 7 + i,
      i9 = 8 + i;

    const x = particleProps[i];
    const y = particleProps[i2];
    const n = noise3D(x * xOff, y * yOff, tick * zOff) * noiseSteps * TAU;
    const smoothing = 1 - Math.pow(0.5, dtFrames);
    const vx = lerp(particleProps[i3], Math.cos(n), smoothing);
    const vy = lerp(particleProps[i4], Math.sin(n), smoothing);
    let life = particleProps[i5];
    const ttl = particleProps[i6];
    const speed = particleProps[i7];
    const x2 = x + vx * speed * dtFrames;
    const y2 = y + vy * speed * dtFrames;
    const radius = particleProps[i8];
    const hue = particleProps[i9];

    if (ctx) {
      drawParticle(x, y, x2, y2, life, ttl, radius, hue, ctx);
    }

    life += dtFrames;

    particleProps[i] = x2;
    particleProps[i2] = y2;
    particleProps[i3] = vx;
    particleProps[i4] = vy;
    particleProps[i5] = life;

    if (checkBounds(x, y, canvas) || life > ttl) {
      initParticle(i);
    }
  }

  function drawParticle(
    x: number,
    y: number,
    x2: number,
    y2: number,
    life: number,
    ttl: number,
    radius: number,
    hue: number,
    ctx: CanvasRenderingContext2D,
  ) {
    const alpha = fadeInOut(life, ttl);

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x2, y2);
    ctx.lineWidth = radius * 3;
    ctx.strokeStyle = `hsla(${hue},100%,60%,${0.3 * alpha})`;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x2, y2);
    ctx.lineWidth = radius;
    ctx.strokeStyle = `hsla(${hue},100%,70%,${alpha})`;
    ctx.stroke();
  }

  function checkBounds(x: number, y: number, canvas: HTMLCanvasElement) {
    return x > canvas.width || x < 0 || y > canvas.height || y < 0;
  }

  function resize(canvas: HTMLCanvasElement, container: HTMLElement) {
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    center[0] = 0.5 * canvas.width;
    center[1] = 0.5 * canvas.height;
  }

  useMountEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !container || !ctx) return;

    resize(canvas, container);
    initParticles();

    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      let staticRedrawFrame: number | undefined;
      const staticResizeObserver = new ResizeObserver(() => {
        if (staticRedrawFrame) window.cancelAnimationFrame(staticRedrawFrame);
        staticRedrawFrame = window.requestAnimationFrame(() => {
          resize(canvas, container);
          initParticles();
          drawStaticFrame(canvas, ctx);
        });
      });
      staticResizeObserver.observe(container);
      return () => {
        if (staticRedrawFrame) window.cancelAnimationFrame(staticRedrawFrame);
        staticResizeObserver.disconnect();
      };
    }

    const resizeObserver = new ResizeObserver(() => {
      const wasEmpty = canvas.width === 0 || canvas.height === 0;
      resize(canvas, container);
      if (wasEmpty && canvas.width > 0 && canvas.height > 0) {
        initParticles();
      }
    });
    resizeObserver.observe(container);

    const intersectionObserver = new IntersectionObserver((entries) => {
      const entry = entries[entries.length - 1];
      if (entry.isIntersecting) {
        startLoop();
      } else {
        stopLoop();
      }
    });
    intersectionObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
      stopLoop();
    };
  });

  return (
    <div className={cn("relative h-full w-full", props.containerClassName)}>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        ref={containerRef}
        className="absolute inset-0 z-0 flex h-full w-full items-center justify-center bg-transparent"
      >
        <canvas ref={canvasRef}></canvas>
      </motion.div>

      <div className={cn("relative z-10", props.className)}>
        {props.children}
      </div>
    </div>
  );
}
