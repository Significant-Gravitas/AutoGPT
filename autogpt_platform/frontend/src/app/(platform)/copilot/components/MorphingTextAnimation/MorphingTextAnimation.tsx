import { cn } from "@/lib/utils";
import { AnimatePresence, LazyMotion, domAnimation, m } from "framer-motion";

interface Props {
  text: string;
  className?: string;
}

export function MorphingTextAnimation({ text, className }: Props) {
  const letters = text.split("");

  return (
    <LazyMotion features={domAnimation}>
      <div className={cn(className)}>
        <AnimatePresence mode="popLayout" initial={false}>
          <m.div key={text} className="whitespace-nowrap">
            <m.span className="inline-flex overflow-hidden">
              {letters.map((char, index) => (
                // eslint-disable-next-line react/no-array-index-key
                <m.span
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
                  transition={{
                    delay: 0.015 * index,
                    type: "spring",
                    bounce: 0.5,
                  }}
                  className="inline-block"
                >
                  {char === " " ? "\u00A0" : char}
                </m.span>
              ))}
            </m.span>
          </m.div>
        </AnimatePresence>
      </div>
    </LazyMotion>
  );
}
