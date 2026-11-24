"use client";

import { motion } from "framer-motion";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Confetti } from "@/components/molecules/Confetti/Confetti";
import { cn } from "@/lib/utils";
import type { IconSvgElement } from "@hugeicons/react";
import { duration, ease, spring } from "@/components/styles/motion";

// The rings that pulse out from the icon: an effect, not a UI transition.
const RING_PULSE_DURATION = 1.6;

interface Hero {
  title: string;
  description: string;
  Icon: IconSvgElement;
  pulse: string;
  gradient: string;
}

interface Props {
  hero: Hero;
  showCelebration: boolean;
  showConfetti: boolean;
  shouldReduceMotion: boolean;
}

export function ReviewHero({
  hero,
  showCelebration,
  showConfetti,
  shouldReduceMotion,
}: Props) {
  return (
    <>
      {showConfetti && !shouldReduceMotion ? (
        <Confetti
          options={{
            particleCount: 50,
            spread: 70,
            startVelocity: 35,
            ticks: 120,
            origin: { y: 0.3 },
          }}
        />
      ) : null}

      <div className="relative flex items-center justify-center">
        {showCelebration && !shouldReduceMotion ? (
          <>
            <motion.span
              aria-hidden
              initial={{ opacity: 0.5, scale: 0.6 }}
              animate={{ opacity: 0, scale: 1.6 }}
              transition={{
                duration: RING_PULSE_DURATION,
                ease: "easeOut",
                repeat: 3,
                repeatDelay: 0.4,
              }}
              className={cn(
                "absolute inline-block size-24 rounded-full",
                hero.pulse,
              )}
            />
            <motion.span
              aria-hidden
              initial={{ opacity: 0.4, scale: 0.7 }}
              animate={{ opacity: 0, scale: 1.4 }}
              transition={{
                duration: RING_PULSE_DURATION,
                ease: "easeOut",
                repeat: 3,
                repeatDelay: 0.4,
                delay: 0.5,
              }}
              className={cn(
                "absolute inline-block size-24 rounded-full",
                hero.pulse,
              )}
            />
          </>
        ) : null}
        <motion.div
          initial={
            shouldReduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.7 }
          }
          animate={{ opacity: 1, scale: 1 }}
          transition={
            shouldReduceMotion
              ? { duration: 0 }
              : { type: "spring", ...spring.gentle, delay: 0.05 }
          }
          className={cn(
            "relative flex size-20 items-center justify-center rounded-full bg-gradient-to-br shadow-drop shadow-brand/45",
            hero.gradient,
          )}
        >
          <span
            aria-hidden
            className="absolute inset-1 rounded-full bg-surface/10"
          />
          <motion.span
            initial={
              shouldReduceMotion ? { opacity: 0 } : { scale: 0.4, opacity: 0 }
            }
            animate={{ scale: 1, opacity: 1 }}
            transition={
              shouldReduceMotion
                ? { duration: 0 }
                : {
                    type: "spring",
                    ...spring.gentle,
                    delay: 0.18,
                  }
            }
            className="text-action-foreground"
          >
            <Icon icon={hero.Icon} size={36} />
          </motion.span>
        </motion.div>
      </div>

      <motion.div
        initial={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: duration.base, ease: ease.out, delay: 0.12 }}
        className="mt-5 flex max-w-md flex-col items-center gap-2 px-2 text-center"
      >
        <Text
          variant="lead"
          weight="medium"
          as="h2"
          data-testid="view-agent-name"
        >
          {hero.title}
        </Text>
        <Text variant="body" tone="secondary">
          {hero.description}
        </Text>
      </motion.div>
    </>
  );
}
