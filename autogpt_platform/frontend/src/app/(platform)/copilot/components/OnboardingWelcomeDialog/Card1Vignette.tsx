"use client";

import { GlassOrb } from "@/app/(no-navbar)/onboarding/steps/BrainDumpStep/components/GlassOrb/GlassOrb";
import { SMALL_ORB_PARAMS } from "../OnboardingIntroCard/OnboardingIntroCard";
import {
  ArrowUpIcon,
  ChartBarIcon,
  CheckIcon,
  ClockIcon,
  PlusIcon,
} from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useEffect, useState } from "react";

const PROMPT = "Send me a competitor report every Monday 9am";

// Three weeks of day cells scrolling under a fixed playhead — every time
// a Monday lands on it, a report drops into the tray. Cell pitch must
// match the w-8 + gap-1 layout below (2rem + 0.25rem).
const DAY_LABELS = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"];
const WEEKS = 3;
const WEEK_PITCH_REM = 7 * 2.25;

// Beat timeline (ms): type + send → the schedule chip snaps onto the ask
// and toggles on → the week strip time-lapses, dropping a report each
// Monday → the toast dips in from the top and leaves on its own.
const SENT_AT = 2000;
const CHIP_AT = 2400;
const TOGGLE_AT = 2750;
const STRIP_AT = 2950;
const WEEK_AT = [3800, 4800];
const DROPS_AT = [3250, 4300, 5300];
const TOAST_IN_AT = 5800;
const TOAST_OUT_AT = 8300;

// One-shot "ask once, get it forever" vignette for the Meet AutoPilot
// card, mapping its copy beat for beat: it does the work (reports), ask
// once / on a schedule (chip + toggle), delivers while you do something
// else (the time-lapse). The component unmounts when the card advances,
// so mounting is the single play; reduced motion renders the final frame.
//
// Layout note: motion elements own the CSS `transform`, so centering
// lives on static wrapper divs and the motion elements animate inside.
export function Card1Vignette() {
  const prefersReducedMotion = useReducedMotion();
  const done = Boolean(prefersReducedMotion);
  const [typed, setTyped] = useState(done ? PROMPT : "");
  const [isSent, setIsSent] = useState(done);
  const [showChip, setShowChip] = useState(done);
  const [isToggled, setIsToggled] = useState(done);
  const [showStrip, setShowStrip] = useState(done);
  const [weekIndex, setWeekIndex] = useState(done ? WEEKS - 1 : 0);
  const [reportCount, setReportCount] = useState(done ? 3 : 0);
  const [isToastVisible, setIsToastVisible] = useState(false);

  useEffect(() => {
    if (prefersReducedMotion) return;

    const typeInterval = setInterval(
      () => {
        setTyped((current) => {
          if (current.length >= PROMPT.length) {
            clearInterval(typeInterval);
            return current;
          }
          return PROMPT.slice(0, current.length + 1);
        });
      },
      1800 / (PROMPT.length + 6),
    );

    const timers = [
      setTimeout(() => setIsSent(true), SENT_AT),
      setTimeout(() => setShowChip(true), CHIP_AT),
      setTimeout(() => setIsToggled(true), TOGGLE_AT),
      setTimeout(() => setShowStrip(true), STRIP_AT),
      ...WEEK_AT.map((at, index) =>
        setTimeout(() => setWeekIndex(index + 1), at),
      ),
      ...DROPS_AT.map((at, index) =>
        setTimeout(() => setReportCount(index + 1), at),
      ),
      setTimeout(() => setIsToastVisible(true), TOAST_IN_AT),
      setTimeout(() => setIsToastVisible(false), TOAST_OUT_AT),
    ];
    return () => {
      clearInterval(typeInterval);
      timers.forEach(clearTimeout);
    };
  }, [prefersReducedMotion]);

  return (
    <div className="relative h-full w-full overflow-hidden">
      {/* Phase A: the ask, typed and sent. */}
      <div className="absolute left-1/2 top-1/2 z-[5] w-[78%] -translate-x-1/2 -translate-y-1/2">
        <AnimatePresence>
          {!isSent && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -16, scale: 0.96 }}
              transition={{ duration: 0.35, ease: [0, 0, 0.2, 1] }}
              className="rounded-2xl border border-zinc-100 bg-white p-4 shadow-lg"
            >
              <div className="min-h-[3.25rem] text-left text-sm text-zinc-800">
                {typed}
                <motion.span
                  animate={{ opacity: [1, 0] }}
                  transition={{ duration: 0.7, repeat: Infinity }}
                  className="ml-0.5 inline-block h-4 w-[2px] translate-y-0.5 bg-violet-500"
                />
              </div>
              <div className="mt-2 flex items-center justify-between">
                <span className="flex h-7 w-7 items-center justify-center rounded-full bg-zinc-100 text-zinc-500">
                  <PlusIcon size={14} />
                </span>
                <motion.span
                  animate={
                    typed.length >= PROMPT.length
                      ? { scale: [1, 0.97, 1] }
                      : undefined
                  }
                  transition={{
                    duration: 0.25,
                    delay: 0.15,
                    ease: [0, 0, 0.2, 1],
                  }}
                  className="flex h-8 w-8 items-center justify-center rounded-full bg-violet-600 text-white shadow-sm"
                >
                  <ArrowUpIcon size={16} weight="bold" />
                </motion.span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* The ask pinned top, with the schedule chip snapped onto it. */}
      {isSent && (
        <div className="absolute left-1/2 top-4 z-[5] flex w-[72%] -translate-x-1/2 flex-col items-end gap-1.5">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: [0, 0, 0.2, 1] }}
            className="w-fit max-w-full rounded-2xl rounded-br-md border border-violet-200 bg-gradient-to-br from-[#f3edff] to-[#e4d4ff] px-3.5 py-2.5 text-left text-xs leading-snug text-[#3b1e75] shadow-sm"
          >
            {PROMPT}
          </motion.div>
          <AnimatePresence>
            {showChip && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95, y: -4 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ duration: 0.25, ease: [0, 0, 0.2, 1] }}
                className="flex items-center gap-2 rounded-full border border-zinc-100 bg-white px-3 py-2 shadow-md"
              >
                <ClockIcon
                  size={14}
                  weight="duotone"
                  className="text-violet-600"
                />
                <span className="whitespace-nowrap text-xs font-medium text-zinc-800">
                  Every Mon · 9:00
                </span>
                <span
                  className={
                    isToggled
                      ? "flex h-4 w-7 items-center rounded-full bg-violet-500 transition-colors"
                      : "flex h-4 w-7 items-center rounded-full bg-zinc-200 transition-colors"
                  }
                >
                  <motion.span
                    animate={{ x: isToggled ? 13 : 2 }}
                    transition={{ type: "spring", stiffness: 400, damping: 32 }}
                    className="h-3 w-3 rounded-full bg-white shadow"
                  />
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}

      {/* The time-lapse: weeks scroll under a fixed playhead. */}
      <div className="absolute left-1/2 top-[52%] z-10 w-[78%] -translate-x-1/2 -translate-y-1/2">
        <AnimatePresence>
          {showStrip && (
            <motion.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, ease: [0, 0, 0.2, 1] }}
              className="relative"
            >
              {/* Playhead pinned to the first cell's slot. */}
              <div className="absolute left-[1rem] top-[-0.55rem] z-10 -translate-x-1/2 text-[8px] text-violet-500">
                ▼
              </div>
              <div className="overflow-hidden rounded-xl border border-zinc-100 bg-white py-1.5 shadow-md">
                <motion.div
                  initial={false}
                  animate={{ x: `-${weekIndex * WEEK_PITCH_REM}rem` }}
                  transition={{ duration: 0.5, ease: [0.32, 0.72, 0, 1] }}
                  className="flex gap-1 pl-1"
                >
                  {Array.from({ length: WEEKS }, (_, week) =>
                    DAY_LABELS.map((day, dayIndex) => {
                      const isMonday = dayIndex === 0;
                      const isAtPlayhead = isMonday && week === weekIndex;
                      return (
                        <span
                          key={`${week}-${day}`}
                          className={
                            isAtPlayhead
                              ? "flex h-7 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-500 text-xs font-semibold text-white transition-colors"
                              : isMonday
                                ? "flex h-7 w-8 shrink-0 items-center justify-center rounded-lg bg-violet-50 text-xs font-medium text-violet-700 transition-colors"
                                : "flex h-7 w-8 shrink-0 items-center justify-center rounded-lg text-xs font-medium text-zinc-700"
                          }
                        >
                          {day}
                        </span>
                      );
                    }),
                  )}
                </motion.div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* The delivery tray, filling up Monday by Monday. */}
      <div className="absolute bottom-11 left-1/2 z-10 flex -translate-x-1/2 items-center gap-1.5">
        <AnimatePresence>
          {Array.from({ length: reportCount }, (_, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: -12, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.3, ease: [0, 0, 0.2, 1] }}
              className="flex items-center gap-2 rounded-xl border border-zinc-100 bg-white px-3 py-2 shadow-md"
            >
              <ChartBarIcon
                size={15}
                weight="duotone"
                className="text-violet-600"
              />
              <span className="text-xs font-medium text-zinc-700">Report</span>
              <span className="flex h-4 w-4 items-center justify-center rounded-full bg-emerald-500 text-white">
                <CheckIcon size={9} weight="bold" />
              </span>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Toast: dips in from the top, leaves on its own after a second. */}
      <div className="absolute left-1/2 top-3 z-30 -translate-x-1/2">
        <AnimatePresence>
          {isToastVisible && (
            <motion.div
              initial={{ opacity: 0, y: -32 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{
                opacity: 0,
                y: -32,
                transition: { duration: 0.25, ease: [0.4, 0, 1, 1] },
              }}
              transition={{ duration: 0.4, ease: [0.32, 0.72, 0, 1] }}
              className="flex items-center gap-2.5 rounded-full bg-zinc-900 py-2.5 pl-2.5 pr-4 shadow-lg"
            >
              <span className="relative h-6 w-6 shrink-0">
                <GlassOrb params={SMALL_ORB_PARAMS} />
              </span>
              <span className="whitespace-nowrap text-sm font-medium text-white">
                Asked once. Delivered every Monday.
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
