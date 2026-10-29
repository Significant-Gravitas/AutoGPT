"use client";

import { cn } from "@/lib/utils";
import type { CSSProperties } from "react";

// Aave-style glass pane (aave.com/design/building-glass-for-the-web):
// frosted backdrop blur + saturation, gradient tint, rim highlights and a
// heavier frost band around the edge. All knobs exposed for the design lab.
export interface GlassParams {
  frost: number;
  saturation: number;
  tint: number;
  edge: number;
  distortion: number;
  ringWidth: number;
  ringDepth: number;
  ringDark: number;
}

export const DEFAULT_GLASS_PARAMS: GlassParams = {
  frost: 8,
  saturation: 1.4,
  tint: 0.35,
  edge: 0.8,
  distortion: 40,
  ringWidth: 2,
  ringDepth: 6,
  ringDark: 0.45,
};

interface Props {
  params: GlassParams;
  showRim?: boolean;
}

export function GlassSurface({ params, showRim = true }: Props) {
  const { frost, saturation, tint, edge } = params;

  // The knobs arrive at runtime, so they reach the classes as custom
  // properties; the tint and rim are white at a knob-driven opacity.
  return (
    <>
      <div
        data-slot="glass-surface"
        className={cn(
          "backdrop-blur-(--glass-frost) backdrop-saturate-(--glass-saturation) pointer-events-none absolute inset-0 rounded-full",
          "bg-linear-155 from-surface/(--glass-tint-from) via-surface/(--glass-tint-via) via-48% to-surface/(--glass-tint-to)",
          showRim && "border-surface/(--glass-edge) border",
        )}
        style={
          {
            "--glass-frost": `${frost}px`,
            "--glass-saturation": saturation,
            "--glass-tint-from": `${tint * 100}%`,
            "--glass-tint-via": `${tint * 20}%`,
            "--glass-tint-to": `${tint * 60}%`,
            "--glass-edge": `${Math.min(edge, 1) * 100}%`,
            // oxlint-disable-next-line design-system/no-inline-styles -- glass rim highlights (white at knob-driven opacities) and the violet glow rgba(96,64,224,0.18) have no role
            boxShadow: showRim
              ? `inset 0 1px 3px rgba(255,255,255,${Math.min(edge * 1.2, 1)}), inset 0 -12px 28px rgba(255,255,255,${edge * 0.5}), 0 12px 40px rgba(96,64,224,0.18)`
              : "0 12px 40px rgba(96,64,224,0.18)",
          } as CSSProperties
        }
      />
      <div
        data-slot="glass-surface-frost"
        className="backdrop-blur-(--glass-frost-band) pointer-events-none absolute inset-0 rounded-full [mask-image:radial-gradient(closest-side,transparent_50%,black_80%)]"
        style={{ "--glass-frost-band": `${frost * 2.5}px` } as CSSProperties}
      />
    </>
  );
}
