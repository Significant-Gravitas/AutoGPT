import { motion } from "framer-motion";
import type { PropsWithChildren } from "react";

interface MotionProps extends PropsWithChildren {
  className?: string;
  delay?: number;
}

const FadeIn = (props: MotionProps) => (
  <motion.div
    initial={{ opacity: 0, y: -30 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, type: "spring", delay: props.delay ?? 0.3 }}
    {...props}
  >
    {props.children}
  </motion.div>
);

FadeIn.displayName = "FadeOut";
export default FadeIn;
