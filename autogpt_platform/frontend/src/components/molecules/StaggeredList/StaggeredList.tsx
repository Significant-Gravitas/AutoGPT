"use client";

import { cn } from "@/lib/utils";
import { motion, useReducedMotion, type Variants } from "framer-motion";
import { ReactNode } from "react";

type StaggerDirection = "up" | "down" | "left" | "right" | "none";

interface StaggeredListProps {
  /** Array of items to render with staggered animation */
  children: ReactNode[];
  /** Direction items animate from */
  direction?: StaggerDirection;
  /** Distance to travel in pixels */
  distance?: number;
  /** Base duration for each item's animation */
  duration?: number;
  /** Delay between each item's animation start */
  staggerDelay?: number;
  /** Initial delay before first item animates */
  initialDelay?: number;
  /** Whether to trigger animation when element enters viewport */
  viewport?: boolean;
  /** How much of container must be visible to trigger */
  viewportAmount?: number;
  /** Whether animation should only trigger once */
  once?: boolean;
  /** Additional CSS classes for the container */
  className?: string;
  /** Additional CSS classes for each item wrapper */
  itemClassName?: string;
}

function getDirectionOffset(
  direction: StaggerDirection,
  distance: number,
): { x: number; y: number } {
  switch (direction) {
    case "up":
      return { x: 0, y: distance };
    case "down":
      return { x: 0, y: -distance };
    case "left":
      return { x: distance, y: 0 };
    case "right":
      return { x: -distance, y: 0 };
    case "none":
    default:
      return { x: 0, y: 0 };
  }
}

/**
 * Animates a list of children with staggered fade-in effects.
 * Each child appears sequentially with a configurable delay.
 * Respects user's reduced motion preferences.
 */
export function StaggeredList({
  children,
  direction = "up",
  distance = 20,
  duration = 0.4,
  staggerDelay = 0.1,
  initialDelay = 0,
  viewport = true,
  viewportAmount = 0.1,
  once = true,
  className,
  itemClassName,
}: StaggeredListProps) {
  const shouldReduceMotion = useReducedMotion();
  const offset = getDirectionOffset(direction, distance);

  // If user prefers reduced motion, render without animation
  if (shouldReduceMotion) {
    return (
      <div className={className}>
        {children.map((child, index) => (
          <div key={index} className={itemClassName}>
            {child}
          </div>
        ))}
      </div>
    );
  }

  const containerVariants: Variants = {
    hidden: {},
    visible: {
      transition: {
        staggerChildren: staggerDelay,
        delayChildren: initialDelay,
      },
    },
  };

  const itemVariants: Variants = {
    hidden: {
      opacity: 0,
      x: offset.x,
      y: offset.y,
    },
    visible: {
      opacity: 1,
      x: 0,
      y: 0,
      transition: {
        duration,
        ease: [0.25, 0.1, 0.25, 1],
      },
    },
  };

  return (
    <motion.div
      className={cn(className)}
      initial="hidden"
      animate={viewport ? undefined : "visible"}
      whileInView={viewport ? "visible" : undefined}
      viewport={viewport ? { once, amount: viewportAmount } : undefined}
      variants={containerVariants}
    >
      {children.map((child, index) => (
        <motion.div
          key={index}
          className={itemClassName}
          variants={itemVariants}
        >
          {child}
        </motion.div>
      ))}
    </motion.div>
  );
}
