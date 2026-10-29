"use client";

import { ReactNode } from "react";
import { GlassParams, GlassSurface } from "./GlassSurface";

// Orb v2: a stack of distorted gradient ellipses (banana / kidney / egg
// shapes, 0.5 opacity each) drifting over each other, sealed under an
// Aave-style glass pane. The displacement filter bends the blobs the way a
// real lens would (feDisplacementMap, per the Aave glass article).
interface Props {
  params: GlassParams;
  children?: ReactNode;
  showRim?: boolean;
}

export function GlassOrb({ params, children, showRim = true }: Props) {
  return (
    <div data-slot="glass-orb" className="relative h-full w-full" aria-hidden>
      <svg
        data-slot="glass-orb-filter"
        className="absolute h-0 w-0"
        aria-hidden
      >
        <filter data-slot="glass-orb-refraction" id="glass-orb-refraction">
          <feTurbulence
            data-slot="glass-orb-noise"
            type="fractalNoise"
            baseFrequency="0.012 0.012"
            numOctaves="2"
            seed="7"
            result="noise"
          />
          <feDisplacementMap
            data-slot="glass-orb-displacement"
            in="SourceGraphic"
            in2="noise"
            scale={params.distortion}
            xChannelSelector="R"
            yChannelSelector="G"
          />
        </filter>
      </svg>

      <div
        data-slot="glass-orb-clip"
        className="absolute inset-0 overflow-hidden rounded-full"
      >
        {/* TODO(tokens): no role for the orb's blue-violet gradient stops;
            the keyframes live in globals.css under "DS2b keyframes". */}
        <div
          data-slot="glass-orb-blobs"
          className="absolute inset-0 [filter:url(#glass-orb-refraction)]"
        >
          <div
            data-slot="glass-orb-orbit"
            className="absolute inset-0 animate-[glass-orb-spin_12s_linear_infinite] motion-reduce:animate-none"
          >
            <div
              data-slot="glass-orb-blob"
              className="absolute left-[6%] top-[10%] h-[46%] w-[78%] animate-[glass-orb-drift-1_9s_ease-in-out_infinite_alternate] rounded-[78%_22%_58%_42%/30%_72%_28%_70%] bg-[linear-gradient(120deg,#9c43fe,#5b8cff)] opacity-50 blur motion-reduce:animate-none"
            />
          </div>
          <div
            data-slot="glass-orb-orbit"
            className="absolute inset-0 animate-[glass-orb-spin_19s_linear_infinite_reverse] motion-reduce:animate-none"
          >
            <div
              data-slot="glass-orb-blob"
              className="absolute left-[34%] top-[18%] h-[70%] w-[55%] animate-[glass-orb-drift-2_11s_ease-in-out_infinite_alternate] rounded-[30%_70%_25%_75%/65%_28%_72%_35%] bg-[linear-gradient(200deg,#7c5bff,#4062ff)] opacity-50 blur motion-reduce:animate-none"
            />
          </div>
          <div
            data-slot="glass-orb-orbit"
            className="absolute inset-0 animate-[glass-orb-spin_15s_linear_infinite] motion-reduce:animate-none"
          >
            <div
              data-slot="glass-orb-blob"
              className="absolute left-[12%] top-[46%] h-[38%] w-[68%] animate-[glass-orb-drift-3_7.5s_ease-in-out_infinite_alternate] rounded-[85%_15%_70%_30%/20%_80%_25%_75%] bg-[linear-gradient(90deg,#b06bff,#6d7dff)] opacity-50 blur motion-reduce:animate-none"
            />
          </div>
          <div
            data-slot="glass-orb-orbit"
            className="absolute inset-0 animate-[glass-orb-spin_24s_linear_infinite_reverse] motion-reduce:animate-none"
          >
            <div
              data-slot="glass-orb-blob"
              className="absolute left-[8%] top-[30%] h-[58%] w-[44%] animate-[glass-orb-drift-4_13s_ease-in-out_infinite_alternate] rounded-[45%_55%_80%_20%/75%_30%_70%_25%] bg-[linear-gradient(160deg,#8a4dff,#4d8bff)] opacity-50 blur motion-reduce:animate-none"
            />
          </div>
          <div
            data-slot="glass-orb-orbit"
            className="absolute inset-0 animate-[glass-orb-spin_9s_linear_infinite] motion-reduce:animate-none"
          >
            <div
              data-slot="glass-orb-blob"
              className="absolute left-[26%] top-[58%] h-[34%] w-[60%] animate-[glass-orb-drift-5_8.5s_ease-in-out_infinite_alternate] rounded-[20%_80%_35%_65%/80%_25%_75%_20%] bg-[linear-gradient(45deg,#a35bff,#5470ff)] opacity-50 blur motion-reduce:animate-none"
            />
          </div>
        </div>
      </div>

      <GlassSurface params={params} showRim={showRim} />

      {children && (
        <div
          data-slot="glass-orb-content"
          className="pointer-events-none absolute inset-0 flex items-center justify-center"
        >
          {children}
        </div>
      )}
    </div>
  );
}
