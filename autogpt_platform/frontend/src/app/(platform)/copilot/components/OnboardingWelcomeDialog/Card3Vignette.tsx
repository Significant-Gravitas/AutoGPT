"use client";

import { GlassOrb } from "@/app/(no-navbar)/onboarding/steps/BrainDumpStep/components/GlassOrb/GlassOrb";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { SMALL_ORB_PARAMS } from "../OnboardingIntroCard/OnboardingIntroCard";
import { GraduationCapIcon } from "@phosphor-icons/react";
import {
  motion,
  useMotionValueEvent,
  useReducedMotion,
  useTime,
  useTransform,
  type MotionValue,
} from "framer-motion";
import { useState } from "react";

const ORB = { x: 24, y: 50 };

// The knowledge wheel: centered on the stage's right edge so only its
// left half shows, radius in px (the stage is ~448×256, so percent
// coords would squash the circle into an ellipse).
const ORBIT_RADIUS = 140;
// One full revolution — the 8-10s the animation is scripted for.
const ORBIT_SECONDS = 9;

// With 6 chips spaced evenly, one crosses the wall every 1.5s; the first
// (baseAngle 150°) reaches 180° after a quarter slot. The knowledge ball
// launches on that clock so it always leaves as a chip converts.
const CROSSING_SECONDS = ORBIT_SECONDS / 6;
const FIRST_CROSSING_SECONDS = CROSSING_SECONDS / 2;
const BALL_SECONDS = 0.9;

// Raw knowledge orbits up the visible left rim; crossing the horizontal
// line is the moment it's learned and becomes one of the orb's own
// files. The reverse crossing happens off-screen behind the right edge.
const KNOWLEDGE = [
  { raw: "generate voice", file: "voice.md", baseAngle: 30 },
  { raw: "weekly report", file: "reports.md", baseAngle: 90 },
  { raw: "escalation rules", file: "escalation.md", baseAngle: 150 },
  { raw: "brand tone", file: "tone.md", baseAngle: 210 },
  { raw: "customer faqs", file: "faqs.md", baseAngle: 270 },
  { raw: "meeting notes", file: "notes.md", baseAngle: 330 },
];

// One continuous loop for the learns-how-you-operate card: orb on the
// left, the knowledge wheel half off-screen on the right, the learning
// line between them. Reduced motion freezes the wheel on a frame where
// half the knowledge is already learned.
export function Card3Vignette() {
  const prefersReducedMotion = useReducedMotion();
  const done = Boolean(prefersReducedMotion);

  const time = useTime();
  const rotation = useTransform(time, (t) =>
    done ? 0 : (t / (ORBIT_SECONDS * 1000)) * 360,
  );

  return (
    <div className="relative h-full w-full overflow-hidden">
      {/* The learning line: constant thickness, running from beside the
          orb to just past the wheel's crossing point, fading out at both
          ends. Below it knowledge is raw, above it learned. */}
      <motion.div
        aria-hidden
        initial={done ? false : { scaleX: 0, opacity: 0 }}
        animate={{ scaleX: 1, opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.4, ease: [0, 0, 0.2, 1] }}
        style={{ left: `${ORB.x + 8}%`, right: ORBIT_RADIUS - 48 }}
        className="absolute top-1/2 h-0.5 -translate-y-1/2 rounded-full bg-[#5b21b6]/45"
      />

      {/* The ball of knowledge: every time a chip crosses the wall (one
          crossing per ORBIT_SECONDS / chip count), a bead flies from the
          crossing point back to the orb. The container spans orb-center →
          crossing point so the flight is a plain 100% → 0% left sweep. */}
      {!done && (
        <div
          aria-hidden
          className="absolute top-1/2"
          style={{ left: `${ORB.x}%`, right: ORBIT_RADIUS }}
        >
          <motion.span
            initial={{ left: "100%", opacity: 0 }}
            animate={{ left: "0%", opacity: [0, 1, 1, 0] }}
            transition={{
              left: {
                duration: BALL_SECONDS,
                ease: "easeInOut",
                repeat: Infinity,
                repeatDelay: CROSSING_SECONDS - BALL_SECONDS,
                delay: FIRST_CROSSING_SECONDS,
              },
              opacity: {
                duration: BALL_SECONDS,
                times: [0, 0.15, 0.85, 1],
                repeat: Infinity,
                repeatDelay: CROSSING_SECONDS - BALL_SECONDS,
                delay: FIRST_CROSSING_SECONDS,
              },
            }}
            className="absolute flex h-6 w-6 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white shadow-[0_0_10px_rgba(139,92,246,0.8)]"
          >
            <GraduationCapIcon
              size={14}
              weight="duotone"
              className="text-violet-600"
            />
          </motion.span>
        </div>
      )}

      {/* The orb, watching knowledge cross the line. */}
      <div
        className="absolute z-10 -translate-x-1/2 -translate-y-1/2"
        style={{ left: `${ORB.x}%`, top: `${ORB.y}%` }}
      >
        <motion.div
          initial={done ? false : { opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.35, delay: 0.3, ease: [0, 0, 0.2, 1] }}
          className="relative h-20 w-20"
        >
          <GlassOrb params={SMALL_ORB_PARAMS} />
          <AutoGPTLogo
            hideText
            className="absolute left-1/2 top-1/2 z-10 h-9 w-[4.9rem] -translate-x-[77%] -translate-y-1/2 brightness-0 drop-shadow-[0_1px_2px_rgba(0,0,0,0.25)] invert"
          />
        </motion.div>
      </div>

      {/* The knowledge wheel, half out of frame on the right. */}
      <motion.div
        initial={done ? false : { opacity: 0, x: 24 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: 0.5, ease: [0, 0, 0.2, 1] }}
        className="absolute right-0 top-1/2"
      >
        <div
          aria-hidden
          className="absolute rounded-full border border-white/70 bg-white/10"
          style={{
            width: ORBIT_RADIUS * 2,
            height: ORBIT_RADIUS * 2,
            left: -ORBIT_RADIUS,
            top: -ORBIT_RADIUS,
          }}
        />
        {KNOWLEDGE.map((item) => (
          <OrbitingKnowledge
            key={item.file}
            rotation={rotation}
            raw={item.raw}
            file={item.file}
            baseAngle={item.baseAngle}
          />
        ))}
      </motion.div>
    </div>
  );
}

interface OrbitingKnowledgeProps {
  rotation: MotionValue<number>;
  raw: string;
  file: string;
  baseAngle: number;
}

// One chip riding the wheel. Position is driven straight from motion
// values (no re-render per frame); only the raw ⇄ learned flip touches
// React state, keyed so the swap pops.
function OrbitingKnowledge({
  rotation,
  raw,
  file,
  baseAngle,
}: OrbitingKnowledgeProps) {
  const angle = useTransform(
    rotation,
    (r) => ((r + baseAngle) * Math.PI) / 180,
  );
  const x = useTransform(angle, (a) => Math.cos(a) * ORBIT_RADIUS);
  const y = useTransform(angle, (a) => Math.sin(a) * ORBIT_RADIUS);

  const [isLearned, setIsLearned] = useState(
    Math.sin((baseAngle * Math.PI) / 180) < 0,
  );
  useMotionValueEvent(y, "change", (latest) => setIsLearned(latest < 0));

  return (
    <motion.div style={{ x, y }} className="absolute left-0 top-0">
      <div className="-translate-x-1/2 -translate-y-1/2">
        {isLearned ? (
          <motion.span
            key="learned"
            initial={{ scale: 0.6, opacity: 0.3 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.3, ease: [0, 0, 0.2, 1] }}
            className="flex items-center gap-1.5 whitespace-nowrap rounded-lg border border-zinc-100 bg-white px-2.5 py-1.5 shadow-md"
          >
            <span className="font-mono text-xs text-zinc-700">{file}</span>
            <span className="h-2 w-2 rounded-full bg-emerald-500" />
          </motion.span>
        ) : (
          <motion.span
            key="raw"
            initial={{ scale: 0.8, opacity: 0.4 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.3, ease: [0, 0, 0.2, 1] }}
            className="whitespace-nowrap rounded-lg border border-white/80 bg-white/50 px-3 py-1.5 text-xs italic text-zinc-500"
          >
            {raw}
          </motion.span>
        )}
      </div>
    </motion.div>
  );
}
