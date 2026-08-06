"use client";

import { ReactNode } from "react";
import {
  motion,
  type MotionValue,
  useMotionValue,
  useReducedMotion,
  useTransform,
} from "framer-motion";
import { GlassParams, GlassSurface } from "./GlassSurface";
import styles from "./GlassOrb.module.css";

// Orb v2: a stack of distorted gradient ellipses (banana / kidney / egg
// shapes, 0.5 opacity each) drifting over each other, sealed under an
// Aave-style glass pane. The displacement filter bends the blobs the way a
// real lens would (feDisplacementMap, per the Aave glass article).
interface Props {
  params: GlassParams;
  audioLevel?: MotionValue<number>;
  children?: ReactNode;
  showRim?: boolean;
}

export function GlassOrb({
  params,
  audioLevel,
  children,
  showRim = true,
}: Props) {
  const prefersReducedMotion = useReducedMotion();
  const idleLevel = useMotionValue(0);
  const level = audioLevel ?? idleLevel;
  const fillScale = useTransform(level, [0, 1], [1, 1.34]);
  const fillOpacity = useTransform(level, [0, 1], [0.72, 1]);
  const pulseOpacity = useTransform(level, [0, 1], [0.08, 0.58]);

  return (
    <motion.div
      className="relative h-full w-full"
      initial={prefersReducedMotion ? false : { scale: 0.94 }}
      animate={{ scale: 1 }}
      transition={
        prefersReducedMotion
          ? undefined
          : { type: "spring", stiffness: 420, damping: 28, mass: 0.7 }
      }
      aria-hidden
    >
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
        <motion.div
          className="absolute inset-0"
          style={{
            filter: "url(#glass-orb-refraction)",
            scale: fillScale,
            opacity: fillOpacity,
          }}
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
        </motion.div>
        <motion.div
          className="pointer-events-none absolute inset-[12%] rounded-full bg-[radial-gradient(circle,rgba(233,213,255,0.9),rgba(168,85,247,0.22)_45%,transparent_72%)] blur-md"
          style={{ opacity: pulseOpacity, scale: fillScale }}
        />
      </div>

      <GlassSurface params={params} showRim={showRim} />

      {children && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          {children}
        </div>
      )}
    </motion.div>
  );
}
