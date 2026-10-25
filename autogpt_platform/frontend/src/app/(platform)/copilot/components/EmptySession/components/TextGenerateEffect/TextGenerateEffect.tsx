"use client";
import { useEffect, useRef } from "react";
import { motion, stagger, useAnimate } from "motion/react";
import { cn } from "@/lib/utils";

interface Props {
  words: string;
  className?: string;
  filter?: boolean;
  duration?: number;
  /** Seconds to wait before the first word starts revealing. */
  delay?: number;
}

// The reveal runs after the first render and once more after the first
// re-render, then never again: the timing this effect had when it depended on
// `scope.current` (null while rendering the first time, the element after),
// counted in an effect so no ref is read during render.
const REVEAL_RUNS = 2;

export function TextGenerateEffect({
  words,
  className,
  filter = true,
  duration = 0.5,
  delay = 0,
}: Props) {
  const [scope, animate] = useAnimate();
  const revealRuns = useRef(0);
  const wordsArray = words.split(" ");
  useEffect(() => {
    if (revealRuns.current >= REVEAL_RUNS) return;
    revealRuns.current += 1;
    animate(
      "span",
      {
        opacity: 1,
        filter: filter ? "blur(0px)" : "none",
      },
      {
        duration: duration ? duration : 1,
        delay: stagger(0.08, { startDelay: delay }),
      },
    );
  });

  const renderWords = () => {
    return (
      <motion.div ref={scope}>
        {wordsArray.map((word, idx) => {
          return (
            <motion.span
              key={word + idx}
              className="text-black opacity-0 dark:text-white"
              style={{
                filter: filter ? "blur(10px)" : "none",
              }}
            >
              {word}{" "}
            </motion.span>
          );
        })}
      </motion.div>
    );
  };

  return (
    <div className={cn("font-bold", className)}>
      <div className="mt-4">
        <div
          // TODO(tokens): no role holds 24px (text-2xl); the type scale goes 22px (heading-sm) to 28px (heading). SECRT-3152
          className="text-2xl leading-snug tracking-wide text-black dark:text-white"
        >
          {renderWords()}
        </div>
      </div>
    </div>
  );
}
