"use client";

import type { MotionValue } from "framer-motion";
import { type ReactNode, useRef } from "react";
import {
  createStreakField,
  createStreakStore,
  type StreakField,
  type StreakStore,
} from "./streaks";
import { VoiceStreaks } from "./VoiceStreaks";

interface Props {
  /** Light and mid tones the comets are drawn in. */
  colors: [string, string];
  size: number;
  levels: MotionValue<number>[];
  isActive: boolean;
  children: ReactNode;
}

// Wraps an avatar in its voice streaks: one canvas behind it, one in front,
// so the comets orbit the body in depth. The avatar goes in as children.
export function VoiceAura({ colors, size, levels, isActive, children }: Props) {
  const fieldRef = useRef<{ key: string; field: StreakField } | null>(null);
  const key = `${colors[0]}.${colors[1]}.${size}`;
  if (fieldRef.current === null || fieldRef.current.key !== key) {
    fieldRef.current = { key, field: createStreakField(colors, size) };
  }
  const storeRef = useRef<StreakStore | null>(null);
  if (storeRef.current === null) storeRef.current = createStreakStore();

  return (
    <div
      data-testid="voice-aura"
      className="relative flex items-center justify-center"
      style={{ width: size, height: size }}
    >
      <VoiceStreaks
        field={fieldRef.current.field}
        levels={levels}
        isActive={isActive}
        layer="behind"
        store={storeRef.current}
      />
      <div className="relative z-10">{children}</div>
      <VoiceStreaks
        field={fieldRef.current.field}
        levels={levels}
        isActive={isActive}
        layer="front"
        store={storeRef.current}
      />
    </div>
  );
}
