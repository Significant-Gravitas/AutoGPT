import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "framer-motion";

interface Props {
  text: string;
  className?: string;
}

export function MorphingTextAnimation({ text, className }: Props) {
  const letters = text.split("");

  return (
    // min-w-0 + overflow-hidden lets the parent flex item shrink and clip;
    // the inner whitespace-nowrap keeps the per-character animation on one
    // line without forcing the chat column wider than the viewport on long
    // command strings.
    <div className={cn("min-w-0 overflow-hidden", className)}>
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.div
          key={text}
          className="truncate whitespace-nowrap"
        >
          <motion.span className="inline-flex overflow-hidden">
            {letters.map((char, index) => (
              <motion.span
                key={`${text}-${index}`}
                initial={{
                  opacity: 0,
                  y: 8,
                  rotateX: "80deg",
                  filter: "blur(6px)",
                }}
                animate={{
                  opacity: 1,
                  y: 0,
                  rotateX: "0deg",
                  filter: "blur(0px)",
                }}
                exit={{
                  opacity: 0,
                  y: -8,
                  rotateX: "-80deg",
                  filter: "blur(6px)",
                }}
                style={{ willChange: "transform" }}
                transition={{
                  delay: 0.015 * index,
                  type: "spring",
                  bounce: 0.5,
                }}
                className="inline-block"
              >
                {char === " " ? "\u00A0" : char}
              </motion.span>
            ))}
          </motion.span>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
