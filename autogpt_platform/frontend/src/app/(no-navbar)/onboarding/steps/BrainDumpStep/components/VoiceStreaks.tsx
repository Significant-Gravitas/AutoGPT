"use client";

import type { MotionValue } from "framer-motion";
import { useEffect, useRef } from "react";
import {
  envelope,
  headAngle,
  hexToRgb,
  makeStreakPair,
  orbitPoint,
  project,
  advanceStreaks,
  type StreakStore,
  type Streak,
  type StreakField,
  tailSpan,
} from "./streaks";

const SAMPLES = 28;

interface Props {
  field: StreakField;
  levels: MotionValue<number>[];
  isActive: boolean;
  layer: "behind" | "front";
  // Shared between the two layers so both draw the same comets.
  store: StreakStore;
}

// One canvas per depth layer. The behind layer sits under the avatar and the
// front layer over it, so a comet on the far side of its orbit is hidden by
// the body and reappears as it comes round. The front layer also owns the
// spawner so streaks are only created once per frame.
export function VoiceStreaks({ field, levels, isActive, layer, store }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const levelsRef = useRef(levels);
  useEffect(() => {
    levelsRef.current = levels;
  }, [levels]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = field.box * ratio;
    canvas.height = field.box * ratio;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);

    if (!isActive) {
      context.clearRect(0, 0, field.box, field.box);
      if (layer === "front") store.streaks = [];
      return;
    }

    // Opening the mic answers with a wave straight away, before any sound,
    // so the click is seen to land; silence then settles into dots.
    if (layer === "front") {
      store.streaks = makeStreakPair(
        field,
        store.nextId,
        0.8,
        performance.now(),
      );
      store.nextId += 2;
      store.lastTick = 0;
      store.budget = 0;
    }

    let frame = 0;
    const tick = (now: number) => {
      if (layer === "front") {
        const currentLevels = levelsRef.current;
        const loudness = currentLevels.length
          ? currentLevels.reduce((sum, level) => sum + level.get(), 0) /
            currentLevels.length
          : 0;
        advanceStreaks(store, field, loudness, now);
      }
      draw(context, field, store.streaks, now, layer);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [field, isActive, layer, store]);

  return (
    <canvas
      ref={canvasRef}
      data-testid={`voice-streaks-${layer}`}
      aria-hidden
      className="pointer-events-none absolute"
      style={{
        inset: -field.pad,
        width: field.box,
        height: field.box,
        zIndex: layer === "front" ? 20 : 0,
      }}
    />
  );
}

function draw(
  context: CanvasRenderingContext2D,
  field: StreakField,
  streaks: Streak[],
  now: number,
  layer: "behind" | "front",
) {
  context.clearRect(0, 0, field.box, field.box);
  context.lineCap = "round";
  for (const streak of streaks) {
    const progress = (now - streak.bornAt) / streak.durationMs;
    const visibility = envelope(progress);
    if (visibility <= 0) continue;
    const head = headAngle(streak, progress);
    const direction = Math.sign(streak.sweep);
    const tail = head - direction * tailSpan(streak, progress);
    const [fromColor, toColor] = streak.colors.map(hexToRgb);

    let previous = project(field, streak, orbitPoint(streak, tail));
    for (let index = 1; index <= SAMPLES; index++) {
      const along = index / SAMPLES;
      const point = orbitPoint(streak, tail + (head - tail) * along);
      const current = project(field, streak, point);
      const onThisLayer = layer === "front" ? point.z >= 0 : point.z < 0;
      if (onThisLayer) {
        const [r, g, b] = fromColor.map((from, channel) =>
          Math.round(from + (toColor[channel] - from) * along),
        );
        const depthAlpha = layer === "front" ? 1 : 0.8;
        context.strokeStyle = `rgba(${r}, ${g}, ${b}, ${visibility * along * depthAlpha})`;
        context.lineWidth = streak.width * current.scale;
        context.beginPath();
        context.moveTo(previous.x + field.pad, previous.y + field.pad);
        context.lineTo(current.x + field.pad, current.y + field.pad);
        context.stroke();
      }
      previous = current;
    }
  }
}
