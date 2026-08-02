"use client";

import { GlassOrb } from "@/app/(no-navbar)/onboarding/steps/BrainDumpStep/components/GlassOrb/GlassOrb";
import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { SMALL_ORB_PARAMS } from "../OnboardingIntroCard/OnboardingIntroCard";
import { WrenchIcon } from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import Image from "next/image";
import { useEffect, useRef, useState } from "react";
import { CurveLab, type Point } from "./CurveLab";

const ORB = { x: 50, y: 42 };

// The four headline tools, arranged around the orb. Coordinates are
// stage percentages shared by the tiles and the SVG arcs; `cx`/`cy` is
// that arc's quadratic control point, omitted to fall back to the
// symmetric default bow.
const TOOLS: {
  icon: string;
  x: number;
  y: number;
  cx?: number;
  cy?: number;
}[] = [
  { icon: "slack", x: 23.5, y: 26.8, cx: 20.8, cy: 38.7 },
  { icon: "google", x: 78, y: 18, cx: 77.3, cy: 32.7 },
  { icon: "github", x: 32.6, y: 64.3, cx: 28.8, cy: 52.5 },
  { icon: "notion", x: 73, y: 59.6, cx: 75.4, cy: 48.9 },
];

// The "and 40+ more" burst: faint tiles around the edges.
const EXTRA_TOOLS = [
  { icon: "discord", x: 8, y: 42 },
  { icon: "linear", x: 41.6, y: 10.6 },
  { icon: "twitter", x: 62.1, y: 12.4 },
  { icon: "telegram", x: 92, y: 42 },
  { icon: "todoist", x: 11.6, y: 78.5 },
  { icon: "hubspot", x: 36, y: 92 },
  { icon: "airtable", x: 67.2, y: 87.4 },
  { icon: "wordpress", x: 92, y: 85 },
];

// Default bow: how far an arc without its own control point pushes off
// the straight line to the orb, in stage percent.
const CURVINESS = 8;

// Curved wiring: a shallow quadratic bow from each tile to the orb,
// mirrored per side so the four arcs read as one symmetric hub.
function defaultControl(tool: Point, curviness: number): Point {
  const midX = (tool.x + ORB.x) / 2;
  const midY = (tool.y + ORB.y) / 2;
  const dx = ORB.x - tool.x;
  const dy = ORB.y - tool.y;
  const length = Math.hypot(dx, dy) || 1;
  const side = tool.x < ORB.x ? 1 : -1;
  return {
    x: Math.round((midX + (-dy / length) * curviness * side) * 10) / 10,
    y: Math.round((midY + (dx / length) * curviness * side) * 10) / 10,
  };
}

function buildControls(curviness: number): Record<string, Point> {
  return Object.fromEntries(
    TOOLS.map((tool) => [
      tool.icon,
      tool.cx !== undefined && tool.cy !== undefined
        ? { x: tool.cx, y: tool.cy }
        : defaultControl(tool, curviness),
    ]),
  );
}

function curvePath(tool: Point, control: Point) {
  return `M ${tool.x} ${tool.y} Q ${control.x} ${control.y} ${ORB.x} ${ORB.y}`;
}

// Beat timeline (ms): the orb lands → the four tools fly in → all four
// wire up to the orb at once (arcs fade in, dots pop, the orb pulses) →
// the wider ecosystem bursts in faintly.
const ORB_AT = 300;
const TILES_AT = 650;
const CONNECT_AT = 1700;
const BURST_AT = 2700;

// The information beads: every wire carries its own, on its own clock —
// different flight times, rests and starts so the traffic overlaps
// irregularly instead of metronoming. Each SMIL cycle = travel + rest;
// the bead moves for the first fraction and hides parked for the rest.
const BEADS = [
  { travel: 1.0, rest: 0.8, begin: 0.4 },
  { travel: 1.35, rest: 1.6, begin: 1.2 },
  { travel: 0.85, rest: 1.1, begin: 0.8 },
  { travel: 1.15, rest: 2.0, begin: 1.7 },
];

function beadTiming(index: number) {
  const bead = BEADS[index % BEADS.length];
  const cycle = bead.travel + bead.rest;
  const fraction = Math.round((bead.travel / cycle) * 100) / 100;
  return { ...bead, cycle, fraction };
}

// The bead's flight path in stage pixels, orb → provider (the arcs'
// stretched percent space can't host a round circle).
function pixelCurve(
  tool: Point,
  control: Point,
  size: { w: number; h: number },
) {
  const px = (point: Point) =>
    `${(point.x / 100) * size.w} ${(point.y / 100) * size.h}`;
  return `M ${px(ORB)} Q ${px(control)} ${px(tool)}`;
}

// One-shot "the hub" vignette for the works-inside-your-tools card. The
// component unmounts when the card advances, so mounting is the single
// play; reduced motion renders the final frame.
//
// Layout note: motion elements own the CSS `transform`, so centering
// lives on static wrapper divs and the motion elements animate inside.
export function Card2Vignette() {
  const prefersReducedMotion = useReducedMotion();
  const done = Boolean(prefersReducedMotion);
  const [showOrb, setShowOrb] = useState(done);
  const [showTiles, setShowTiles] = useState(done);
  const [isConnected, setIsConnected] = useState(done);
  const [showBurst, setShowBurst] = useState(done);

  // Testing-only curve lab: toggle with the wrench, drag each arc's pen
  // handle (or bow them all with the slider), then "Copy" pastes the
  // TOOLS snippet back into this file. Remove before release.
  const [isDebug, setIsDebug] = useState(false);
  const [curviness, setCurviness] = useState(CURVINESS);
  const [controls, setControls] = useState(() => buildControls(CURVINESS));
  const stageRef = useRef<HTMLDivElement>(null);

  // Stage pixel size, for the beads' pixel-space SVG.
  const [stageSize, setStageSize] = useState<{ w: number; h: number } | null>(
    null,
  );
  useEffect(() => {
    const node = stageRef.current;
    if (!node) return;
    const update = () =>
      setStageSize({ w: node.clientWidth, h: node.clientHeight });
    update();
    const observer = new ResizeObserver(update);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  function handleCurvinessChange(next: number) {
    setCurviness(next);
    setControls(
      Object.fromEntries(
        TOOLS.map((tool) => [tool.icon, defaultControl(tool, next)]),
      ),
    );
  }

  useEffect(() => {
    if (prefersReducedMotion) return;

    const timers = [
      setTimeout(() => setShowOrb(true), ORB_AT),
      setTimeout(() => setShowTiles(true), TILES_AT),
      setTimeout(() => setIsConnected(true), CONNECT_AT),
      setTimeout(() => setShowBurst(true), BURST_AT),
    ];
    return () => timers.forEach(clearTimeout);
  }, [prefersReducedMotion]);

  const orbVisible = showOrb || isDebug;
  const tilesVisible = showTiles || isDebug;
  const burstVisible = showBurst || isDebug;
  const wired = isConnected || isDebug;

  return (
    <div ref={stageRef} className="relative h-full w-full overflow-hidden">
      {/* The wiring: full arcs fading in together. */}
      <svg
        className="absolute inset-0 h-full w-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        fill="none"
      >
        {TOOLS.map((tool, index) => (
          <g key={tool.icon}>
            <motion.path
              d={curvePath(tool, controls[tool.icon])}
              stroke="#8b5cf6"
              strokeWidth="2"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
              initial={{ opacity: 0 }}
              animate={wired ? { opacity: 0.55 } : {}}
              transition={{
                duration: 0.35,
                ease: [0, 0, 0.2, 1],
                delay: done || isDebug ? 0 : index * 0.05,
              }}
            />
          </g>
        ))}
      </svg>

      {/* The information beads: builder-style circles (SMIL animateMotion)
          riding the arcs from inside the orb out to each provider, one at
          a time round-robin. They live in their own pixel-space SVG — the
          arcs' stretched 0-100 viewBox would squash a circle into an
          ellipse — so the stage is measured and the same curves rebuilt
          in px. */}
      {!done && !isDebug && isConnected && stageSize && (
        <svg
          className="absolute inset-0 h-full w-full"
          viewBox={`0 0 ${stageSize.w} ${stageSize.h}`}
          fill="none"
        >
          {TOOLS.map((tool, index) => {
            const timing = beadTiming(index);
            return (
              <circle key={tool.icon} r="4.5" fill="#f5f3ff" opacity="0">
                <animateMotion
                  dur={`${timing.cycle}s`}
                  repeatCount="indefinite"
                  begin={`${timing.begin}s`}
                  path={pixelCurve(tool, controls[tool.icon], stageSize)}
                  keyPoints={`0;1;1`}
                  keyTimes={`0;${timing.fraction};1`}
                  calcMode="spline"
                  keySplines="0.4 0 0.6 1;0 0 1 1"
                />
                <animate
                  attributeName="opacity"
                  values="0;0.95;0.95;0;0"
                  keyTimes={`0;0.02;${timing.fraction - 0.02};${timing.fraction};1`}
                  dur={`${timing.cycle}s`}
                  repeatCount="indefinite"
                  begin={`${timing.begin}s`}
                />
              </circle>
            );
          })}
        </svg>
      )}

      {/* The orb, hub of it all. */}
      <div
        className="absolute z-10 -translate-x-1/2 -translate-y-1/2"
        style={{ left: `${ORB.x}%`, top: `${ORB.y}%` }}
      >
        <AnimatePresence>
          {orbVisible && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{
                opacity: 1,
                scale: isConnected && !done ? [1, 1.05, 1] : 1,
              }}
              transition={{
                opacity: { duration: 0.35, ease: [0, 0, 0.2, 1] },
                scale: { duration: 0.5, ease: [0, 0, 0.2, 1] },
              }}
              className="relative h-16 w-16"
            >
              <GlassOrb params={SMALL_ORB_PARAMS} />
              {/* The mark sits in the right ~46% of the logo's viewBox, so
                  centering needs a 77% shift; brightness-0 + invert
                  flattens the gradient mark to solid white. */}
              <AutoGPTLogo
                hideText
                className="absolute left-1/2 top-1/2 z-10 h-7 w-[3.9rem] -translate-x-[77%] -translate-y-1/2 brightness-0 drop-shadow-[0_1px_2px_rgba(0,0,0,0.25)] invert"
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* The four headline tools. */}
      {tilesVisible &&
        TOOLS.map((tool, index) => (
          <div
            key={tool.icon}
            className="absolute z-10 -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${tool.x}%`, top: `${tool.y}%` }}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              transition={{
                duration: 0.3,
                ease: [0, 0, 0.2, 1],
                delay: done || isDebug ? 0 : index * 0.07,
              }}
              className="relative flex h-11 w-11 items-center justify-center rounded-2xl border border-zinc-100 bg-white shadow-md"
            >
              <Image
                src={`/integrations/${tool.icon}.png`}
                alt=""
                width={24}
                height={24}
                className="pointer-events-none rounded-sm"
              />
              <AnimatePresence>
                {wired && (
                  <motion.span
                    initial={{ opacity: 0, scale: 0.5 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{
                      duration: 0.2,
                      ease: [0, 0, 0.2, 1],
                      delay: done ? 0 : 0.35,
                    }}
                    className="absolute -right-1 -top-1 h-3 w-3 rounded-full border-2 border-white bg-emerald-500"
                  />
                )}
              </AnimatePresence>
            </motion.div>
          </div>
        ))}

      {/* The "and 40+ more" burst around the edges. */}
      {burstVisible &&
        EXTRA_TOOLS.map((tool, index) => (
          <div
            key={tool.icon}
            className="absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${tool.x}%`, top: `${tool.y}%` }}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 0.55, scale: 1 }}
              transition={{
                duration: 0.3,
                ease: [0, 0, 0.2, 1],
                delay: done || isDebug ? 0 : index * 0.05,
              }}
              className="flex h-8 w-8 items-center justify-center rounded-xl border border-zinc-100 bg-white shadow-sm"
            >
              <Image
                src={`/integrations/${tool.icon}.png`}
                alt=""
                width={16}
                height={16}
                className="pointer-events-none rounded-sm"
              />
            </motion.div>
          </div>
        ))}

      {/* Testing-only curve lab controls. Remove before release. */}
      <button
        type="button"
        aria-label="Toggle curve debug"
        onClick={() => setIsDebug((debug) => !debug)}
        className={
          isDebug
            ? "absolute right-3 top-3 z-30 flex h-7 w-7 items-center justify-center rounded-full bg-zinc-900 text-white shadow"
            : "absolute right-3 top-3 z-30 flex h-7 w-7 items-center justify-center rounded-full text-[#5b21b6]/50 transition-colors hover:bg-white/50"
        }
      >
        <WrenchIcon size={14} />
      </button>
      {isDebug && (
        <CurveLab
          stageRef={stageRef}
          orb={ORB}
          tools={TOOLS}
          controls={controls}
          onControlChange={(icon, control) =>
            setControls((current) => ({ ...current, [icon]: control }))
          }
          curviness={curviness}
          onCurvinessChange={handleCurvinessChange}
        />
      )}
    </div>
  );
}
