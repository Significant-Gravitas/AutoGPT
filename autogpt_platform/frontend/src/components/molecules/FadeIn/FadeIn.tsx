"use client";

import { cn } from "@/lib/utils";
import { motion, useReducedMotion, type Variants } from "framer-motion";
import { ReactNode } from "react";

type FadeDirection = "up" | "down" | "left" | "right" | "none";

interface FadeInProps {
  /** Content to animate */
  children: ReactNode;
  /** Direction the content fades in from */
  direction?: FadeDirection;
  /** Distance to travel in pixels (only applies when direction is not "none") */
  distance?: number;
  /** Animation duration in seconds */
  duration?: number;
  /** Delay before animation starts in seconds */
  delay?: number;
  /** Whether to trigger animation when element enters viewport */
  viewport?: boolean;
  /** How much of element must be visible to trigger (0-1) */
  viewportAmount?: number;
  /** Whether animation should only trigger once */
  once?: boolean;
  /** Additional CSS classes */
  className?: string;
  /** HTML element to render as */
  as?: keyof JSX.IntrinsicElements;
}

function getDirectionOffset(
  direction: FadeDirection,
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
 * A fade-in animation wrapper component.
 * Animates children with a fade effect and optional directional slide.
 * Respects user's reduced motion preferences.
 */
export function FadeIn({
  children,
  direction = "up",
  distance = 24,
  duration = 0.5,
  delay = 0,
  viewport = true,
  viewportAmount = 0.2,
  once = true,
  className,
  as = "div",
}: FadeInProps) {
  const shouldReduceMotion = useReducedMotion();
  const offset = getDirectionOffset(direction, distance);

  // If user prefers reduced motion, render without animation
  if (shouldReduceMotion) {
    const Component = as as keyof JSX.IntrinsicElements;
    return <Component className={className}>{children}</Component>;
  }

  const variants: Variants = {
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
        delay,
        ease: [0.25, 0.1, 0.25, 1], // Custom easing for smooth feel
      },
    },
  };

  const MotionComponent = motion[
    as as keyof typeof motion
  ] as typeof motion.div;

  return (
    <MotionComponent
      className={cn(className)}
      initial="hidden"
      animate={viewport ? undefined : "visible"}
      whileInView={viewport ? "visible" : undefined}
      viewport={viewport ? { once, amount: viewportAmount } : undefined}
      variants={variants}
    >
      {children}
    </MotionComponent>
  );
}
