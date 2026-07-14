"use client";

import { motion } from "framer-motion";
import { type Icon as PhosphorIcon } from "@phosphor-icons/react";

import { Text } from "@/components/atoms/Text/Text";
import { Confetti } from "@/components/molecules/Confetti/Confetti";
import { cn } from "@/lib/utils";

interface Hero {
  title: string;
  description: string;
  Icon: PhosphorIcon;
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
  const HeroIcon = hero.Icon;

  return (
    <>
      {showConfetti && !shouldReduceMotion ? (
        <Confetti
          options={{
            particleCount: 80,
            spread: 70,
            startVelocity: 35,
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
                duration: 1.6,
                ease: "easeOut",
                repeat: Infinity,
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
                duration: 1.6,
                ease: "easeOut",
                repeat: Infinity,
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
              : { type: "spring", stiffness: 280, damping: 20, delay: 0.05 }
          }
          className={cn(
            "relative flex size-20 items-center justify-center rounded-full bg-gradient-to-br shadow-[0_8px_24px_-8px_rgba(119,51,245,0.45)]",
            hero.gradient,
          )}
        >
          <span
            aria-hidden
            className="absolute inset-1 rounded-full bg-white/10"
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
                    stiffness: 360,
                    damping: 18,
                    delay: 0.18,
                  }
            }
            className="text-white"
          >
            <HeroIcon size={36} weight="bold" />
          </motion.span>
        </motion.div>
      </div>

      <motion.div
        initial={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.24, ease: "easeOut", delay: 0.12 }}
        className="mt-5 flex max-w-md flex-col items-center gap-2 px-2 text-center"
      >
        <Text
          variant="lead-medium"
          as="h2"
          className="text-textBlack"
          data-testid="view-agent-name"
        >
          {hero.title}
        </Text>
        <Text variant="body" className="text-zinc-600">
          {hero.description}
        </Text>
      </motion.div>
    </>
  );
}
