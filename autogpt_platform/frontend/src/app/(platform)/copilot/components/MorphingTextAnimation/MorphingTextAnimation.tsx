import { cn } from "@/lib/utils";
import { motion, useReducedMotion } from "framer-motion";

interface Props {
  text: string;
  className?: string;
  // Only fade in for live, actively-streaming tool calls. Historical lines
  // (e.g. on page reload) render static so we don't replay an entrance for
  // every past tool call at once.
  animate?: boolean;
}

export function MorphingTextAnimation({
  text,
  className,
  animate = false,
}: Props) {
  const reduceMotion = useReducedMotion();
  const shouldAnimate = animate && !reduceMotion;

  return (
    // No key on `text`: the element stays mounted while the line streams in, so
    // the fade plays once on appear instead of restarting on every token.
    <motion.div
      className={cn("min-w-0 truncate whitespace-nowrap", className)}
      aria-label={text}
      initial={shouldAnimate ? { opacity: 0 } : false}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      {text}
    </motion.div>
  );
}
