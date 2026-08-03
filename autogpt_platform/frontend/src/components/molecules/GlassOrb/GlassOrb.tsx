"use client";

import { ReactNode } from "react";
import { GlassParams, GlassSurface } from "./GlassSurface";
import styles from "./GlassOrb.module.css";

// Orb v2: a stack of distorted gradient ellipses (banana / kidney / egg
// shapes, 0.5 opacity each) drifting over each other, sealed under an
// Aave-style glass pane. The displacement filter bends the blobs the way a
// real lens would (feDisplacementMap, per the Aave glass article).
interface Props {
  params: GlassParams;
  children?: ReactNode;
}

export function GlassOrb({ params, children }: Props) {
  return (
    <div className="relative h-full w-full" aria-hidden>
      <svg className="absolute h-0 w-0" aria-hidden>
        <filter id="glass-orb-refraction">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.012 0.012"
            numOctaves="2"
            seed="7"
            result="noise"
          />
          <feDisplacementMap
            in="SourceGraphic"
            in2="noise"
            scale={params.distortion}
            xChannelSelector="R"
            yChannelSelector="G"
          />
        </filter>
      </svg>

      <div className="absolute inset-0 overflow-hidden rounded-full">
        <div
          className="absolute inset-0"
          style={{ filter: "url(#glass-orb-refraction)" }}
        >
          <div className={`${styles.spinner} ${styles.spinner1}`}>
            <div className={`${styles.blob} ${styles.blob1}`} />
          </div>
          <div className={`${styles.spinner} ${styles.spinner2}`}>
            <div className={`${styles.blob} ${styles.blob2}`} />
          </div>
          <div className={`${styles.spinner} ${styles.spinner3}`}>
            <div className={`${styles.blob} ${styles.blob3}`} />
          </div>
          <div className={`${styles.spinner} ${styles.spinner4}`}>
            <div className={`${styles.blob} ${styles.blob4}`} />
          </div>
          <div className={`${styles.spinner} ${styles.spinner5}`}>
            <div className={`${styles.blob} ${styles.blob5}`} />
          </div>
        </div>
      </div>

      <GlassSurface params={params} />

      {children && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          {children}
        </div>
      )}
    </div>
  );
}
