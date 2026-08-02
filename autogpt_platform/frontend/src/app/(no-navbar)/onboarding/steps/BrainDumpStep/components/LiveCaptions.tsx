"use client";

import {
  animate,
  motion,
  useMotionValue,
  useReducedMotion,
} from "framer-motion";
import { useLayoutEffect, useRef } from "react";
import { type CaptionsEngine, useLiveCaptions } from "./useLiveCaptions";

interface Props {
  isRecording: boolean;
  audioStream: MediaStream | null;
  engine?: CaptionsEngine;
}

// Must match the `gap-2` between words below.
const WORD_GAP_PX = 8;

// Proof that something is listening. The words come from the browser's own
// speech recogniser and are never sent anywhere — the real transcript is
// produced server-side from the recording. Where the API is missing
// (Firefox, some Safari builds) this degrades to a level meter, which
// carries the same "we can hear you" signal without pretending to
// transcribe.
export function LiveCaptions({ isRecording, audioStream, engine }: Props) {
  const { words, level, isSpeechSupported } = useLiveCaptions({
    isRecording,
    audioStream,
    engine,
  });
  const prefersReducedMotion = useReducedMotion();
  const lineRef = useRef<HTMLDivElement>(null);
  const lineWidthRef = useRef(0);
  const wordWidthsRef = useRef(new Map<string, number>());
  const x = useMotionValue(0);

  // The line is pinned to the right edge of the box, so the browser lays out
  // every new word as an instant leftward shove of everything already there.
  // Measuring how much the line grew on the right (ignoring words trimmed
  // off the invisible left tail, which right-anchoring makes free) and
  // paying that shove back through one spring on the whole line turns it
  // into a marquee glide: the newest word slides in from past the right
  // edge while the oldest drift left and dissolve into the mask.
  useLayoutEffect(() => {
    const line = lineRef.current;
    if (!line) {
      lineWidthRef.current = 0;
      wordWidthsRef.current = new Map();
      x.jump(0);
      return;
    }
    const widths = new Map<string, number>();
    for (const child of line.children) {
      const id = (child as HTMLElement).dataset.wordId;
      if (id) widths.set(id, (child as HTMLElement).offsetWidth);
    }
    let trimmedWidth = 0;
    for (const [id, width] of wordWidthsRef.current) {
      if (!widths.has(id)) trimmedWidth += width + WORD_GAP_PX;
    }
    const lineWidth = line.scrollWidth;
    const grownBy = lineWidth - lineWidthRef.current + trimmedWidth;
    wordWidthsRef.current = widths;
    lineWidthRef.current = lineWidth;
    if (prefersReducedMotion || grownBy === 0) return;
    x.jump(x.get() + grownBy);
    animate(x, 0, { type: "spring", stiffness: 240, damping: 36 });
  }, [words, prefersReducedMotion, x]);

  if (!isRecording) return null;

  if (!isSpeechSupported) {
    return (
      <div
        className="flex h-10 items-end gap-1"
        aria-hidden
        data-testid="brain-dump-level-meter"
      >
        {[0, 1, 2, 3, 4, 5, 6].map((bar) => (
          <span
            key={bar}
            className="w-1 rounded-full bg-purple-300 transition-[height] duration-150"
            style={{
              height: `${8 + level * 24 * (bar % 3 === 0 ? 1 : 0.6)}px`,
            }}
          />
        ))}
      </div>
    );
  }

  return (
    <div className="relative h-10 w-[min(30rem,calc(100vw-3rem))] shrink-0 overflow-hidden text-xl [-webkit-mask-image:linear-gradient(to_right,transparent_0%,black_25%,black_100%)] [mask-image:linear-gradient(to_right,transparent_0%,black_25%,black_100%)]">
      <motion.div
        ref={lineRef}
        style={{ x }}
        className="absolute inset-y-0 right-0 flex items-center gap-2 whitespace-nowrap"
      >
        {words.map((word) => (
          <motion.span
            key={word.id}
            data-word-id={word.id}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.25, ease: "easeOut" }}
            className="text-zinc-500"
          >
            {word.text}
          </motion.span>
        ))}
      </motion.div>
    </div>
  );
}
