import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "framer-motion";

interface Props {
  text: string;
  className?: string;
}

export function MorphingTextAnimation({ text, className }: Props) {
  const letters = text.split("");

  return (
    <div className={cn(className)}>
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.div key={text} className="whitespace-nowrap">
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
