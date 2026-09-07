"use client";

import { motion, useReducedMotion } from "framer-motion";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { HomeEmptyIllustration } from "./HomeEmptyIllustration";

const EASE_OUT_QUINT = [0.22, 1, 0.36, 1] as const;

interface Props {
  title: string;
  description: string;
  action?: { href: string; label: string };
  className?: string;
}

/** The panels' shared empty state, in the library page's idiom: a ghost
 *  illustration, then the copy and an optional action, each fading up in
 *  turn. Reduced-motion users get a plain fade. */
export function HomeTileEmpty({
  title,
  description,
  action,
  className,
}: Props) {
  const shouldReduceMotion = useReducedMotion();

  function fadeUp(delay: number) {
    if (shouldReduceMotion) {
      return {
        initial: { opacity: 0 },
        animate: { opacity: 1 },
        transition: { duration: 0.2, delay: 0 },
      };
    }
    return {
      initial: { opacity: 0, y: 6 },
      animate: { opacity: 1, y: 0 },
      transition: { duration: 0.35, ease: EASE_OUT_QUINT, delay },
    };
  }

  return (
    <div
      className={cn(
        "flex min-h-[12rem] flex-1 flex-col items-center justify-center gap-5 px-4 py-8 text-center",
        className,
      )}
    >
      <motion.div {...fadeUp(0)}>
        <HomeEmptyIllustration />
      </motion.div>
      <div className="flex max-w-xs flex-col items-center gap-1">
        <motion.div {...fadeUp(0.22)}>
          <Text variant="large-medium" tone="primary">
            {title}
          </Text>
        </motion.div>
        <motion.div {...fadeUp(0.3)}>
          <Text variant="body" tone="muted" className="text-pretty">
            {description}
          </Text>
        </motion.div>
      </div>
      {action ? (
        <motion.div {...fadeUp(0.38)}>
          <Button
            as="NextLink"
            href={action.href}
            variant="secondary"
            size="xs"
          >
            {action.label}
          </Button>
        </motion.div>
      ) : null}
    </div>
  );
}
