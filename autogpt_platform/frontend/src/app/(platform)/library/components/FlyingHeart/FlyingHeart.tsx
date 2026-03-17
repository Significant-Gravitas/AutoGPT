"use client";

import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import { HeartIcon } from "@phosphor-icons/react";
import { useEffect, useState } from "react";

interface FlyingHeartProps {
  startPosition: { x: number; y: number } | null;
  targetPosition: { x: number; y: number } | null;
  onAnimationComplete: () => void;
}

export function FlyingHeart({
  startPosition,
  targetPosition,
  onAnimationComplete,
}: FlyingHeartProps) {
  const [isVisible, setIsVisible] = useState(false);
  const shouldReduceMotion = useReducedMotion();

  useEffect(() => {
    if (startPosition && targetPosition) {
      setIsVisible(true);
    }
  }, [startPosition, targetPosition]);

  if (!startPosition || !targetPosition) return null;

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.div
          className="pointer-events-none fixed z-50"
          initial={{
            x: startPosition.x,
            y: startPosition.y,
            scale: 1,
            opacity: 1,
          }}
          animate={{
            x: shouldReduceMotion ? targetPosition.x : targetPosition.x,
            y: shouldReduceMotion ? targetPosition.y : targetPosition.y,
            scale: 0.5,
            opacity: 0,
          }}
          exit={{ opacity: 0 }}
          transition={
            shouldReduceMotion
              ? { duration: 0 }
              : {
                  type: "spring",
                  damping: 20,
                  stiffness: 200,
                  duration: 0.5,
                }
          }
          onAnimationComplete={() => {
            setIsVisible(false);
            onAnimationComplete();
          }}
        >
          <HeartIcon
            size={24}
            weight="fill"
            className="text-red-500 drop-shadow-md"
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
